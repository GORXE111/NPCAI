#if UNITY_EDITOR
using UnityEditor;

namespace NPCLLM.Editor
{
    [InitializeOnLoad]
    public static class AutoRefresh
    {
        private static double _lastRefreshTime;
        private const double REFRESH_INTERVAL = 5.0;

        static AutoRefresh()
        {
            EditorApplication.update += OnUpdate;
        }

        private static void OnUpdate()
        {
            if (EditorApplication.isPlaying || EditorApplication.isCompiling)
                return;
            double now = EditorApplication.timeSinceStartup;
            if (now - _lastRefreshTime < REFRESH_INTERVAL)
                return;
            _lastRefreshTime = now;
            AssetDatabase.Refresh(ImportAssetOptions.Default);
        }
    }
}
#endif
