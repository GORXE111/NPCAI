using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace NPCAI.VN
{
    /// <summary>
    /// Bottom-of-screen dialogue box with speaker name and typewriter-effect text.
    /// </summary>
    public class DialogueBox : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private TMP_Text speakerLabel;
        [SerializeField] private TMP_Text bodyText;
        [SerializeField] private GameObject continueIndicator;
        [SerializeField] private CanvasGroup canvasGroup;

        [Header("Behavior")]
        [SerializeField] private float charsPerSecond = 50f;
        [SerializeField] private float fadeInTime = 0.15f;

        Coroutine typing;
        bool isTyping;
        public bool IsTyping => isTyping;

        public void Hide()
        {
            if (canvasGroup) canvasGroup.alpha = 0;
            if (continueIndicator) continueIndicator.SetActive(false);
        }

        public void Show()
        {
            if (canvasGroup) canvasGroup.alpha = 1;
        }

        public void DisplayDialogue(string speaker, string text, System.Action onComplete = null)
        {
            if (typing != null) StopCoroutine(typing);
            Show();
            if (speakerLabel) speakerLabel.text = speaker ?? "";
            typing = StartCoroutine(Typewriter(text, onComplete));
        }

        public void SkipTypewriter()
        {
            if (typing != null) { StopCoroutine(typing); typing = null; }
            isTyping = false;
            if (continueIndicator) continueIndicator.SetActive(true);
        }

        IEnumerator Typewriter(string text, System.Action onComplete)
        {
            isTyping = true;
            if (continueIndicator) continueIndicator.SetActive(false);
            if (bodyText) bodyText.text = "";
            float interval = 1f / Mathf.Max(charsPerSecond, 1f);
            for (int i = 0; i < text.Length; i++)
            {
                if (bodyText) bodyText.text += text[i];
                yield return new WaitForSeconds(interval);
            }
            isTyping = false;
            if (continueIndicator) continueIndicator.SetActive(true);
            onComplete?.Invoke();
        }
    }
}
