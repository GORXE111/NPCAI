using System.Collections;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

namespace NPCLLM.UI
{
    /// <summary>
    /// World-space speech bubble + name label for top-down 2D view.
    /// Both bubble and name face the camera (flat on XZ plane for top-down).
    /// </summary>
    public class SpeechBubble : MonoBehaviour
    {
        [Header("Settings")]
        [SerializeField] private float charsPerSecond = 25f;
        [SerializeField] private float displayDuration = 5f;
        [SerializeField] private float bubbleOffsetY = 0.05f;
        [SerializeField] private float bubbleOffsetZ = 1.2f;
        [SerializeField] private float nameOffsetZ = 0.8f;

        private GameObject _bubbleRoot;
        private Canvas _bubbleCanvas;
        private TextMeshProUGUI _bubbleText;
        private Image _bubbleBg;
        private RectTransform _bubbleBgRect;

        private GameObject _nameRoot;
        private Canvas _nameCanvas;
        private TextMeshProUGUI _nameText;

        private Coroutine _typewriterCoroutine;
        private Coroutine _hideCoroutine;
        private string _fullText;
        private bool _isShowing;

        public bool IsShowing => _isShowing;

        private void Awake()
        {
            CreateNameLabel();
            CreateBubbleUI();
            _bubbleRoot.SetActive(false);
        }

        private void LateUpdate()
        {
            // Top-down camera: rotate canvases to face straight down (camera looks along -Y)
            // So canvases should lie flat on XZ plane, text facing up
            Quaternion faceUp = Quaternion.Euler(90f, 0f, 0f);

            if (_nameCanvas != null)
                _nameCanvas.transform.rotation = faceUp;

            if (_bubbleCanvas != null && _isShowing)
                _bubbleCanvas.transform.rotation = faceUp;
        }

        public void SetName(string npcName, Color color)
        {
            if (_nameText != null)
            {
                _nameText.text = npcName;
                _nameText.color = color;
            }
        }

        public void ShowText(string text)
        {
            StopAllBubbleCoroutines();
            _fullText = text;
            _bubbleRoot.SetActive(true);
            _isShowing = true;
            _bubbleText.text = "";
            _typewriterCoroutine = StartCoroutine(TypewriterRoutine());
        }

        public void ShowThinking()
        {
            StopAllBubbleCoroutines();
            _bubbleRoot.SetActive(true);
            _isShowing = true;
            _typewriterCoroutine = StartCoroutine(ThinkingRoutine());
        }

        public void Hide()
        {
            StopAllBubbleCoroutines();
            _bubbleRoot.SetActive(false);
            _isShowing = false;
            _bubbleText.text = "";
        }

        private void StopAllBubbleCoroutines()
        {
            if (_typewriterCoroutine != null) StopCoroutine(_typewriterCoroutine);
            if (_hideCoroutine != null) StopCoroutine(_hideCoroutine);
            _typewriterCoroutine = null;
            _hideCoroutine = null;
        }

        private IEnumerator TypewriterRoutine()
        {
            _bubbleText.text = "";
            for (int i = 0; i < _fullText.Length; i++)
            {
                _bubbleText.text = _fullText.Substring(0, i + 1);
                yield return new WaitForSeconds(1f / charsPerSecond);
            }
            _hideCoroutine = StartCoroutine(AutoHideRoutine());
        }

        private IEnumerator ThinkingRoutine()
        {
            string[] frames = { ".", "..", "..." };
            int idx = 0;
            while (true)
            {
                _bubbleText.text = frames[idx % frames.Length];
                idx++;
                yield return new WaitForSeconds(0.5f);
            }
        }

        private IEnumerator AutoHideRoutine()
        {
            float duration = Mathf.Max(displayDuration, _fullText.Length * 0.08f);
            yield return new WaitForSeconds(duration);
            Hide();
        }

        // ── UI Creation ──────────────────────────

        private void CreateNameLabel()
        {
            _nameRoot = new GameObject("NameLabel");
            _nameRoot.transform.SetParent(transform);
            // Offset in Z (forward in top-down = up on screen)
            _nameRoot.transform.localPosition = new Vector3(0, bubbleOffsetY, nameOffsetZ);
            _nameRoot.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);

            _nameCanvas = _nameRoot.AddComponent<Canvas>();
            _nameCanvas.renderMode = RenderMode.WorldSpace;
            _nameCanvas.sortingOrder = 99;
            _nameRoot.AddComponent<CanvasScaler>().dynamicPixelsPerUnit = 100;

            var canvasRect = _nameRoot.GetComponent<RectTransform>();
            canvasRect.sizeDelta = new Vector2(200f, 30f);
            canvasRect.localScale = new Vector3(0.01f, 0.01f, 0.01f);

            var textGO = new GameObject("Text");
            textGO.transform.SetParent(_nameRoot.transform, false);
            _nameText = textGO.AddComponent<TextMeshProUGUI>();
            _nameText.fontSize = 20f;
            _nameText.color = Color.white;
            _nameText.alignment = TextAlignmentOptions.Center;
            _nameText.fontStyle = FontStyles.Bold;
            _nameText.outlineWidth = 0.3f;
            _nameText.outlineColor = Color.black;
            _nameText.textWrappingMode = TextWrappingModes.NoWrap;

            var textRect = textGO.GetComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = Vector2.zero;
            textRect.offsetMax = Vector2.zero;
        }

        private void CreateBubbleUI()
        {
            _bubbleRoot = new GameObject("SpeechBubble");
            _bubbleRoot.transform.SetParent(transform);
            _bubbleRoot.transform.localPosition = new Vector3(0, bubbleOffsetY, bubbleOffsetZ);
            _bubbleRoot.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);

            _bubbleCanvas = _bubbleRoot.AddComponent<Canvas>();
            _bubbleCanvas.renderMode = RenderMode.WorldSpace;
            _bubbleCanvas.sortingOrder = 100;
            _bubbleRoot.AddComponent<CanvasScaler>().dynamicPixelsPerUnit = 100;

            var canvasRect = _bubbleRoot.GetComponent<RectTransform>();
            canvasRect.sizeDelta = new Vector2(300f, 120f);
            canvasRect.localScale = new Vector3(0.01f, 0.01f, 0.01f);

            // Background
            var bgGO = new GameObject("Background");
            bgGO.transform.SetParent(_bubbleRoot.transform, false);
            _bubbleBg = bgGO.AddComponent<Image>();
            _bubbleBg.color = new Color(0.05f, 0.05f, 0.05f, 0.9f);

            _bubbleBgRect = bgGO.GetComponent<RectTransform>();
            _bubbleBgRect.anchorMin = Vector2.zero;
            _bubbleBgRect.anchorMax = Vector2.one;
            _bubbleBgRect.offsetMin = Vector2.zero;
            _bubbleBgRect.offsetMax = Vector2.zero;

            // Text
            var textGO = new GameObject("Text");
            textGO.transform.SetParent(bgGO.transform, false);
            _bubbleText = textGO.AddComponent<TextMeshProUGUI>();
            _bubbleText.fontSize = 12f;
            _bubbleText.color = Color.white;
            _bubbleText.alignment = TextAlignmentOptions.TopLeft;
            _bubbleText.textWrappingMode = TextWrappingModes.Normal;
            _bubbleText.overflowMode = TextOverflowModes.Overflow;

            var textRect = textGO.GetComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = new Vector2(8f, 6f);
            textRect.offsetMax = new Vector2(-8f, -6f);
        }
    }
}
