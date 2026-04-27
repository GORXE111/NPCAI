using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NPCAI.Agent;
using NPCLLM.LLM;

namespace NPCAI.VN
{
    /// <summary>
    /// Top-level orchestrator. Owns the SceneState, ToolRegistry, UI references,
    /// and drives the player-choice → LLM → tool-execution loop.
    /// </summary>
    public class VisualNovelManager : MonoBehaviour
    {
        [Header("UI References")]
        [SerializeField] private DialogueBox dialogueBox;
        [SerializeField] private ChoicePanel choicePanel;
        [SerializeField] private BackgroundPanel backgroundPanel;
        [SerializeField] private AudioManager audioManager;

        [Header("Character Slots")]
        [SerializeField] private CharacterPortrait slotLeft;
        [SerializeField] private CharacterPortrait slotCenter;
        [SerializeField] private CharacterPortrait slotRight;

        [Header("LLM")]
        [SerializeField] private OllamaClient ollamaClient;

        [Header("Persona")]
        [SerializeField, TextArea(3, 12)] private string personaPrompt = "";

        [Header("Opening Scene")]
        [SerializeField] private string openingLocation = "Whirling-in-Rags Hostel";
        [SerializeField] private string openingBg = "bg_hostel_lobby";
        [SerializeField] private string openingBgm = "bgm_off_white";
        [SerializeField] private string openingActor = "Kim Kitsuragi";
        [SerializeField] private string openingChoice = "Detective approaches.";

        public SceneState Scene { get; private set; }
        public ToolRegistry Tools { get; private set; }

        bool busy;

        void Awake()
        {
            Scene = new SceneState
            {
                locationName = openingLocation,
                backgroundId = openingBg,
                bgmId = openingBgm,
            };
            Tools = new ToolRegistry();
            BuiltinToolHandlers.RegisterAll(this, Tools);
        }

        IEnumerator Start()
        {
            // Apply opening state
            if (backgroundPanel) backgroundPanel.SetBackground(openingBg);
            if (audioManager) audioManager.PlayBgm(openingBgm);
            ShowCharacter(openingActor, "neutral", "center");
            Scene.charactersOnScreen.Add(new CharacterOnScreen { actor = openingActor, slot = "center" });

            yield return new WaitForSeconds(0.5f);

            // Kick the agent for the first turn
            yield return StartCoroutine(RequestNpcTurn(openingChoice));
        }

        public void OnPlayerChoicePicked(int idx, string text)
        {
            if (busy) return;
            Scene.AddTurn("Detective", text);
            // Show in dialogue box briefly
            if (dialogueBox) dialogueBox.DisplayDialogue("Detective", text);
            StartCoroutine(RequestNpcTurn(text));
        }

        IEnumerator RequestNpcTurn(string playerInput)
        {
            busy = true;
            var msgs = PromptBuilder.Build(
                string.IsNullOrEmpty(personaPrompt) ? null : personaPrompt,
                Scene, playerInput, Tools);

            string raw = null;
            string err = null;
            bool done = false;
            ollamaClient.SendChat(msgs,
                onSuccess: r => { raw = r; done = true; },
                onError: e => { err = e; done = true; });

            float timeout = 60f;
            float t = 0;
            while (!done && t < timeout) { t += Time.deltaTime; yield return null; }

            if (err != null) Debug.LogError($"[VN] LLM error: {err}");
            if (string.IsNullOrEmpty(raw))
            {
                if (dialogueBox) dialogueBox.DisplayDialogue("System", "(no response)");
                busy = false;
                yield break;
            }

            var resp = AgentResponse.Parse(raw);
            if (!resp.valid)
            {
                Debug.LogWarning($"[VN] Parse warning: {resp.parseError}\nRaw: {raw}");
            }

            // 1. Show dialogue
            if (!string.IsNullOrEmpty(resp.dialogue))
            {
                string speaker = Scene.charactersOnScreen.Count > 0
                    ? Scene.charactersOnScreen[0].actor : "Kim Kitsuragi";
                Scene.AddTurn(speaker, resp.dialogue);
                bool typingDone = false;
                if (dialogueBox)
                    dialogueBox.DisplayDialogue(speaker, resp.dialogue, () => typingDone = true);
                else typingDone = true;
                while (!typingDone) yield return null;
            }

            // 2. Execute tool calls in order, awaiting each
            foreach (var call in resp.toolCalls)
            {
                bool toolDone = false;
                Tools.Execute(call, _ => toolDone = true);
                while (!toolDone) yield return null;
            }

            busy = false;
        }

        // Convenience helpers used by tool handlers --------------------

        public void ShowCharacter(string actor, string emotion, string slot)
        {
            var portrait = SlotByName(slot);
            if (portrait == null) portrait = slotCenter;
            portrait.SetCharacter(actor, emotion);
            UpsertCharacter(actor, emotion, slot);
        }

        public void HideCharacter(string actor)
        {
            foreach (var p in new[] { slotLeft, slotCenter, slotRight })
                if (p && p.CurrentActor == actor) p.Hide();
            Scene.charactersOnScreen.RemoveAll(c => c.actor == actor);
        }

        public void SetExpression(string actor, string emotion)
        {
            foreach (var p in new[] { slotLeft, slotCenter, slotRight })
                if (p && p.CurrentActor == actor) p.SetEmotion(emotion);
            foreach (var c in Scene.charactersOnScreen)
                if (c.actor == actor) c.emotion = emotion;
        }

        public void SetBackground(string bgId)
        {
            if (backgroundPanel) backgroundPanel.SetBackground(bgId);
            Scene.backgroundId = bgId;
        }

        public void PlayBgm(string trackId) { if (audioManager) audioManager.PlayBgm(trackId); Scene.bgmId = trackId; }
        public void StopBgm() { if (audioManager) audioManager.StopBgm(); Scene.bgmId = null; }
        public void PlaySfx(string sfxId) { if (audioManager) audioManager.PlaySfx(sfxId); }

        public void PresentChoices(IList<string> options)
        {
            if (choicePanel == null || options == null || options.Count == 0) return;
            choicePanel.Present(options, OnPlayerChoicePicked);
        }

        public void Narrate(string text)
        {
            Scene.AddTurn("Narrator", text);
            if (dialogueBox) dialogueBox.DisplayDialogue("", text);
        }

        public void EndScene(string nextSceneId)
        {
            Debug.Log($"[VN] EndScene -> {nextSceneId}");
            Scene.sceneId = nextSceneId ?? "end";
            // Production: load scenario file
        }

        CharacterPortrait SlotByName(string name)
        {
            if (string.IsNullOrEmpty(name)) return slotCenter;
            switch (name.ToLowerInvariant())
            {
                case "left":   return slotLeft;
                case "right":  return slotRight;
                default:       return slotCenter;
            }
        }

        void UpsertCharacter(string actor, string emotion, string slot)
        {
            for (int i = 0; i < Scene.charactersOnScreen.Count; i++)
            {
                if (Scene.charactersOnScreen[i].actor == actor)
                {
                    Scene.charactersOnScreen[i].emotion = emotion;
                    Scene.charactersOnScreen[i].slot = slot;
                    return;
                }
            }
            Scene.charactersOnScreen.Add(new CharacterOnScreen
            { actor = actor, emotion = emotion, slot = slot });
        }
    }
}
