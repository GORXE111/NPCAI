using System.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace NPCAI.VN
{
    /// <summary>
    /// Full-screen background image. Sprites loaded from Resources/Backgrounds/.
    /// </summary>
    public class BackgroundPanel : MonoBehaviour
    {
        [SerializeField] private Image background;
        [SerializeField] private Image overlay;        // for crossfade
        [SerializeField] private float fadeTime = 0.5f;
        [SerializeField] private string folder = "Backgrounds";

        public string CurrentBg { get; private set; }

        public void SetBackground(string bgId)
        {
            if (CurrentBg == bgId) return;
            var sprite = Resources.Load<Sprite>($"{folder}/{bgId}");
            if (sprite == null && background != null)
            {
                Debug.LogWarning($"[BackgroundPanel] Missing bg: {bgId}, using placeholder color");
                background.color = ColorFromString(bgId);
                background.sprite = null;
                CurrentBg = bgId;
                return;
            }
            if (background)
            {
                background.sprite = sprite;
                background.color = Color.white;
            }
            CurrentBg = bgId;
        }

        Color ColorFromString(string s)
        {
            int seed = 0;
            foreach (var c in s ?? "") seed = seed * 31 + c;
            var rng = new System.Random(seed);
            return new Color((float)rng.NextDouble() * 0.4f,
                             (float)rng.NextDouble() * 0.4f,
                             (float)rng.NextDouble() * 0.4f);
        }
    }
}
