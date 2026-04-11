#if UNITY_EDITOR
using UnityEditor;
using System.IO;

namespace NPCLLM.Editor
{
    [InitializeOnLoad]
    public static class AutoPlayMode
    {
        private static readonly string EnterFlag = Path.Combine(
            Path.GetDirectoryName(UnityEngine.Application.dataPath), ".playmode_enter");
        private static readonly string ExitFlag = Path.Combine(
            Path.GetDirectoryName(UnityEngine.Application.dataPath), ".playmode_exit");

        static AutoPlayMode()
        {
            EditorApplication.update += CheckFlags;
        }

        private static void CheckFlags()
        {
            if (File.Exists(EnterFlag))
            {
                File.Delete(EnterFlag);
                if (!EditorApplication.isPlaying)
                {
                    UnityEngine.Debug.Log("[AutoPlayMode] Entering Play Mode via flag file");
                    EditorApplication.EnterPlaymode();
                }
            }
            if (File.Exists(ExitFlag))
            {
                File.Delete(ExitFlag);
                if (EditorApplication.isPlaying)
                {
                    UnityEngine.Debug.Log("[AutoPlayMode] Exiting Play Mode via flag file");
                    EditorApplication.ExitPlaymode();
                }
            }
        }
    }
}
#endif
