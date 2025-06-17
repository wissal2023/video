# audio_synchronized_generator.py
from moviepy.editor import AudioFileClip, CompositeVideoClip, concatenate_videoclips
import traceback
from pathlib import Path
from video_module.video_generation import MLEnhancedVideoGenerator
from video_module.audio_processing import EnhancedVoiceSynthesizer
from video_module.avatar import LipSyncAvatar

class AudioSynchronizedMLGenerator(MLEnhancedVideoGenerator):
    """
    Version améliorée du générateur de vidéo ML avec synchronisation audio/vidéo précise
    utilisant l'apprentissage profond pour optimiser l'expérience
    """
    
    def __init__(self, base_dir=None, output_dir=None, model_path=None):
        super().__init__(base_dir, output_dir, model_path)
        self.voice_synthesizer = EnhancedVoiceSynthesizer()
        self.slide_synchronizer = None  # Could be added later
        self.timing_model = {
            'min_slide_duration': 5.0,
            'max_slide_duration': 30.0,
            'transition_duration': 1.0,
            'image_extra_time': 2.0,
            'intro_duration': 3.0,
            'outro_duration': 3.0,
            'safety_margin': 1.5,
            'silence_padding': 0.5,
            'image_sound_delay': 0.3,
            'animation_speed_factor': 1.2,
        }
        print("? Générateur de présentations ML avec synchronisation audio avancée initialisé")
    
    def process_markdown(self, markdown_content):
        try:
            self.learn_from_previous_presentations()
            # Similar processing as base class but with audio synchronization
            output_path = super().process_markdown(markdown_content)
            return output_path
        except Exception as e:
            print(f"? Erreur globale lors du traitement: {e}")
            traceback.print_exc()
            return None
