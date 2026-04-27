using System.Collections;
using UnityEngine;

namespace NPCAI.VN
{
    /// <summary>
    /// BGM + SFX. Cross-fades BGM tracks. Audio loaded from Resources/BGM/ and Resources/SFX/.
    /// </summary>
    public class AudioManager : MonoBehaviour
    {
        [SerializeField] private AudioSource bgmA;
        [SerializeField] private AudioSource bgmB;
        [SerializeField] private AudioSource sfx;

        [SerializeField] private float crossfadeTime = 1.0f;
        [SerializeField] private float bgmVolume = 0.7f;

        public string CurrentBgm { get; private set; }

        Coroutine fading;
        bool useA = true;

        public void PlayBgm(string trackId)
        {
            if (CurrentBgm == trackId) return;
            var clip = Resources.Load<AudioClip>($"BGM/{trackId}");
            if (clip == null)
            {
                Debug.LogWarning($"[AudioManager] BGM not found: {trackId}");
                CurrentBgm = trackId; // remember even if missing
                return;
            }
            CurrentBgm = trackId;
            if (fading != null) StopCoroutine(fading);
            fading = StartCoroutine(Crossfade(clip));
        }

        public void StopBgm()
        {
            if (fading != null) StopCoroutine(fading);
            if (bgmA) bgmA.Stop();
            if (bgmB) bgmB.Stop();
            CurrentBgm = null;
        }

        public void PlaySfx(string sfxId)
        {
            var clip = Resources.Load<AudioClip>($"SFX/{sfxId}");
            if (clip == null) { Debug.LogWarning($"[AudioManager] SFX not found: {sfxId}"); return; }
            if (sfx) sfx.PlayOneShot(clip);
        }

        IEnumerator Crossfade(AudioClip newClip)
        {
            var fadeOut = useA ? bgmA : bgmB;
            var fadeIn = useA ? bgmB : bgmA;
            useA = !useA;
            if (fadeIn == null || fadeOut == null) yield break;

            fadeIn.clip = newClip;
            fadeIn.volume = 0;
            fadeIn.loop = true;
            fadeIn.Play();
            float t = 0;
            while (t < crossfadeTime)
            {
                t += Time.deltaTime;
                float k = t / crossfadeTime;
                fadeIn.volume = Mathf.Lerp(0, bgmVolume, k);
                fadeOut.volume = Mathf.Lerp(bgmVolume, 0, k);
                yield return null;
            }
            fadeOut.Stop();
            fadeOut.clip = null;
        }
    }
}
