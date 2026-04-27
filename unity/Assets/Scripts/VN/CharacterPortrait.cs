using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace NPCAI.VN
{
    /// <summary>
    /// One on-screen character portrait. Supports expression swap with crossfade.
    /// Sprites are looked up from a resource library by "{actor}_{emotion}".
    /// </summary>
    public class CharacterPortrait : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private Image portrait;
        [SerializeField] private CanvasGroup canvasGroup;

        [Header("Settings")]
        [SerializeField] private float fadeTime = 0.25f;
        [SerializeField] private string spriteFolder = "Portraits";   // Resources/Portraits/Kim_neutral.png ...

        public string CurrentActor { get; private set; }
        public string CurrentEmotion { get; private set; }

        Coroutine fading;

        public void Hide()
        {
            if (canvasGroup) canvasGroup.alpha = 0;
            CurrentActor = null;
            CurrentEmotion = null;
        }

        public void SetCharacter(string actor, string emotion = "neutral")
        {
            if (CurrentActor == actor && CurrentEmotion == emotion) return;
            string spriteName = $"{actor}_{emotion}";
            var sprite = Resources.Load<Sprite>($"{spriteFolder}/{spriteName}");
            if (sprite == null)
            {
                // Try fallback to neutral
                sprite = Resources.Load<Sprite>($"{spriteFolder}/{actor}_neutral");
            }
            if (sprite == null)
            {
                Debug.LogWarning($"[CharacterPortrait] Sprite not found: {spriteName}");
                if (portrait) portrait.color = GetPlaceholderColor(actor);
                if (portrait) portrait.sprite = null;
            }
            else
            {
                if (portrait)
                {
                    portrait.sprite = sprite;
                    portrait.color = Color.white;
                }
            }
            CurrentActor = actor;
            CurrentEmotion = emotion;
            if (fading != null) StopCoroutine(fading);
            fading = StartCoroutine(FadeTo(1f));
        }

        public void SetEmotion(string emotion)
        {
            if (string.IsNullOrEmpty(CurrentActor)) return;
            SetCharacter(CurrentActor, emotion);
        }

        IEnumerator FadeTo(float target)
        {
            if (canvasGroup == null) yield break;
            float start = canvasGroup.alpha;
            float t = 0;
            while (t < fadeTime)
            {
                t += Time.deltaTime;
                canvasGroup.alpha = Mathf.Lerp(start, target, t / fadeTime);
                yield return null;
            }
            canvasGroup.alpha = target;
        }

        // Stable per-actor placeholder color when sprite missing.
        static readonly Dictionary<string, Color> _palette = new Dictionary<string, Color>();
        static Color GetPlaceholderColor(string actor)
        {
            if (_palette.TryGetValue(actor, out var c)) return c;
            int seed = 0;
            foreach (var ch in actor) seed = seed * 31 + ch;
            var rng = new System.Random(seed);
            c = new Color((float)rng.NextDouble() * 0.6f + 0.3f,
                          (float)rng.NextDouble() * 0.6f + 0.3f,
                          (float)rng.NextDouble() * 0.6f + 0.3f);
            _palette[actor] = c;
            return c;
        }
    }
}
