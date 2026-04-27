using System;
using System.Collections.Generic;
using UnityEngine;

namespace NPCAI.VN
{
    /// <summary>
    /// Current visual-novel scene state, used both internally
    /// and serialized into the LLM prompt as context.
    /// </summary>
    [Serializable]
    public class SceneState
    {
        public string sceneId = "intro";
        public string locationName = "Whirling-in-Rags Hostel";
        public string backgroundId = "bg_hostel_lobby";
        public string bgmId = "bgm_off_white";
        public List<CharacterOnScreen> charactersOnScreen = new List<CharacterOnScreen>();
        public List<DialogueTurn> recentTurns = new List<DialogueTurn>();
        public int turnsRetained = 8;

        public void AddTurn(string speaker, string text)
        {
            recentTurns.Add(new DialogueTurn { speaker = speaker, text = text });
            if (recentTurns.Count > turnsRetained)
                recentTurns.RemoveAt(0);
        }

        public string Summarize()
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("[Current Scene]");
            sb.AppendLine($"Location: {locationName}");
            sb.AppendLine($"Background: {backgroundId}");
            sb.AppendLine($"BGM: {bgmId}");
            if (charactersOnScreen.Count > 0)
            {
                sb.Append("Characters present: ");
                for (int i = 0; i < charactersOnScreen.Count; i++)
                {
                    if (i > 0) sb.Append(", ");
                    sb.Append($"{charactersOnScreen[i].actor} ({charactersOnScreen[i].emotion})");
                }
                sb.AppendLine();
            }
            if (recentTurns.Count > 0)
            {
                sb.AppendLine("\n[Recent dialogue]");
                foreach (var t in recentTurns) sb.AppendLine($"{t.speaker}: {t.text}");
            }
            return sb.ToString();
        }
    }

    [Serializable]
    public class CharacterOnScreen
    {
        public string actor;
        public string emotion = "neutral";
        public string slot = "center";   // left | center | right
    }

    [Serializable]
    public class DialogueTurn
    {
        public string speaker;
        public string text;
    }
}
