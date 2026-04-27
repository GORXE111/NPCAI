using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace NPCAI.VN
{
    /// <summary>
    /// Vertical stack of choice buttons. Populated by `present_choices` tool.
    /// Reports back the chosen index/text via callback.
    /// </summary>
    public class ChoicePanel : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private Transform buttonContainer;
        [SerializeField] private Button buttonPrefab;
        [SerializeField] private CanvasGroup canvasGroup;

        readonly List<Button> _spawned = new List<Button>();
        Action<int, string> _onPick;

        public bool IsShowing => canvasGroup != null && canvasGroup.alpha > 0.5f;

        void Awake() { Hide(); }

        public void Hide()
        {
            if (canvasGroup) { canvasGroup.alpha = 0; canvasGroup.blocksRaycasts = false; canvasGroup.interactable = false; }
        }

        public void Show()
        {
            if (canvasGroup) { canvasGroup.alpha = 1; canvasGroup.blocksRaycasts = true; canvasGroup.interactable = true; }
        }

        public void Present(IList<string> options, Action<int, string> onPick)
        {
            ClearButtons();
            _onPick = onPick;
            for (int i = 0; i < options.Count; i++)
            {
                int idx = i;
                string txt = options[i];
                if (buttonPrefab == null || buttonContainer == null) continue;
                var btn = Instantiate(buttonPrefab, buttonContainer);
                var label = btn.GetComponentInChildren<TMP_Text>();
                if (label) label.text = $"{i + 1}. {txt}";
                btn.onClick.AddListener(() => OnPicked(idx, txt));
                _spawned.Add(btn);
            }
            Show();
        }

        void OnPicked(int idx, string txt)
        {
            Hide();
            var cb = _onPick;
            _onPick = null;
            ClearButtons();
            cb?.Invoke(idx, txt);
        }

        void ClearButtons()
        {
            foreach (var b in _spawned) if (b != null) Destroy(b.gameObject);
            _spawned.Clear();
        }
    }
}
