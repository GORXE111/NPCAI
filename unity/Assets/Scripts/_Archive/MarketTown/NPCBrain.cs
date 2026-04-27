using System;
using System.Collections.Generic;
using UnityEngine;
using NPCLLM.LLM;
using NPCLLM.UI;

namespace NPCLLM.NPC
{
    [Serializable]
    public class NPCPersonality
    {
        public string npcName;
        public string occupation;
        public string personality;
        public string speechStyle;
        public string backstory;
        public List<string> knowledge;

        public string ToPrompt()
        {
            string knowledgeStr = knowledge != null && knowledge.Count > 0
                ? "\nYour knowledge: " + string.Join("; ", knowledge)
                : "";
            return "You are " + npcName + ", a " + occupation + " in a medieval market town.\n" +
                   "Personality: " + personality + "\n" +
                   "Speech style: " + speechStyle + "\n" +
                   "Backstory: " + backstory + knowledgeStr + "\n" +
                   "Stay in character at all times. Never mention AI, models, or break the fourth wall.\n" +
                   "Reply concisely in 1-3 sentences.";
        }
    }

    public class NPCBrain : MonoBehaviour
    {
        [Header("Identity")]
        [SerializeField] private string npcId;
        [SerializeField] private NPCPersonality personality;

        [Header("State")]
        [SerializeField] private bool isBusy;
        [SerializeField] private string lastResponse;
        [SerializeField] private float lastResponseTime;

        private NPCMemory _memory;
        private OllamaClient _ollamaClient;
        private SpeechBubble _bubble;
        private string _savePath;

        private static readonly Dictionary<string, string> NpcNames = new Dictionary<string, string>();

        public string NpcId => npcId;
        public string NpcName => personality != null ? personality.npcName : npcId;
        public bool IsBusy => isBusy;
        public NPCPersonality Personality => personality;

        private void Awake()
        {
            if (string.IsNullOrEmpty(npcId))
                npcId = gameObject.name.ToLowerInvariant().Replace(" ", "_");

            _savePath = System.IO.Path.Combine(Application.persistentDataPath, "npc_data");
            _memory = new NPCMemory(npcId, _savePath);

            SocialGraph.Instance.Initialize(_savePath);
            SocialGraph.Instance.RegisterNPC(npcId);

            NpcNames[npcId] = personality != null ? personality.npcName : npcId;

            // Add speech bubble
            _bubble = gameObject.AddComponent<SpeechBubble>();
        }

        /// <summary>
        /// Set name label color (called by TownManager after personality is set).
        /// </summary>
        public void SetNameColor(Color color)
        {
            if (_bubble != null && personality != null)
                _bubble.SetName(personality.npcName, Color.white);
        }

        private void Start()
        {
            _ollamaClient = FindFirstObjectByType<OllamaClient>();
            if (_ollamaClient == null)
                Debug.LogError("[NPCBrain] " + npcId + ": No OllamaClient found in scene!");
        }

        public void TalkTo(string playerMessage, Action<string> onResponse)
        {
            if (isBusy)
            {
                onResponse?.Invoke("*is busy*");
                return;
            }
            if (_ollamaClient == null)
            {
                onResponse?.Invoke("*cannot think right now*");
                return;
            }

            isBusy = true;
            var messages = BuildMessages(playerMessage, "Player");
            _memory.AddShortTerm("Player said: " + playerMessage);

            // Show thinking bubble
            if (_bubble != null) _bubble.ShowThinking();

            float startTime = Time.realtimeSinceStartup;

            _ollamaClient.SendChat(messages,
                response =>
                {
                    lastResponse = response;
                    lastResponseTime = Time.realtimeSinceStartup - startTime;
                    isBusy = false;

                    _memory.AddShortTerm("I replied: " + response);
                    _memory.AddMemory(
                        "Player said '" + playerMessage + "', I replied '" + response + "'",
                        "episodic", 0.6f);

                    // Show response in bubble
                    if (_bubble != null) _bubble.ShowText(response);

                    Debug.Log("[NPCBrain] " + npcId + " (" + lastResponseTime.ToString("F1") + "s): " + response);
                    onResponse?.Invoke(response);
                },
                error =>
                {
                    isBusy = false;
                    if (_bubble != null) _bubble.Hide();
                    Debug.LogError("[NPCBrain] " + npcId + " error: " + error);
                    onResponse?.Invoke("*mumbles incoherently*");
                });
        }

        public void TalkToNPC(string fromNpcId, string fromNpcName, string message, Action<string> onResponse)
        {
            if (isBusy)
            {
                onResponse?.Invoke("");
                return;
            }

            isBusy = true;
            var messages = BuildMessages(message, fromNpcName);

            if (_bubble != null) _bubble.ShowThinking();

            _ollamaClient.SendChat(messages,
                response =>
                {
                    isBusy = false;
                    lastResponse = response;

                    _memory.AddMemory(fromNpcName + " told me: " + message, "episodic", 0.7f);
                    SocialGraph.Instance.UpdateRelation(npcId, fromNpcId, 0.05f, 0.02f);
                    SocialGraph.Instance.ShareMemory(fromNpcId, npcId, message);

                    if (_bubble != null) _bubble.ShowText(response);

                    onResponse?.Invoke(response);
                },
                error =>
                {
                    isBusy = false;
                    if (_bubble != null) _bubble.Hide();
                    onResponse?.Invoke("");
                });
        }

        /// <summary>
        /// Show a speech bubble for the initiator's opening line (no LLM needed).
        /// </summary>
        public void ShowSpeech(string text)
        {
            if (_bubble != null) _bubble.ShowText(text);
        }

        private List<ChatMessage> BuildMessages(string userMessage, string speakerName)
        {
            var messages = new List<ChatMessage>();

            string systemPrompt = personality.ToPrompt();

            string memorySummary = _memory.GetMemorySummary(5);
            if (memorySummary != "No significant memories.")
                systemPrompt += "\n\nYour recent memories:\n" + memorySummary;

            string socialSummary = SocialGraph.Instance.GetSocialSummary(npcId, NpcNames);
            if (socialSummary != "No significant relationships." && socialSummary != "No social connections.")
                systemPrompt += "\n\nYour relationships:\n" + socialSummary;

            messages.Add(new ChatMessage { role = "system", content = systemPrompt });

            foreach (string mem in _memory.GetShortTerm())
            {
                if (mem.StartsWith("Player said:") || mem.StartsWith(speakerName + " said:"))
                    messages.Add(new ChatMessage { role = "user", content = mem.Substring(mem.IndexOf(':') + 2) });
                else if (mem.StartsWith("I replied:"))
                    messages.Add(new ChatMessage { role = "assistant", content = mem.Substring(11) });
            }

            messages.Add(new ChatMessage { role = "user", content = userMessage });

            return messages;
        }

        public void SetPersonality(NPCPersonality p)
        {
            personality = p;
            NpcNames[npcId] = p.npcName;
        }

        public NPCMemory Memory => _memory;
    }
}
