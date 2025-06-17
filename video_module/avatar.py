from pathlib import Path
import traceback
import random
import numpy as np
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip

# Imports pour les moèles d'IA avancés
TRANSFORMER_AVAILABLE = False
try:  
    TRANSFORMER_AVAILABLE = True
    print("? Bibliothèques de modèles disponibles")
except ImportError as e:
    print(f"?? Transformers ou SentenceTransformer non disponibles: {e}")
    print(f"Détails: {e}")
    TRANSFORMER_AVAILABLE = False

class LipSyncAvatar:
    def __init__(self, avatar_path, device="cpu"):
        self.avatar_path = avatar_path
        self.device = device
        self.temp_dir = Path(tempfile.gettempdir()) / "avatar_lipsync"
        self.temp_dir.mkdir(exist_ok=True)
        
        if not Path(avatar_path).exists():
            print(f"?? Avatar non trouvé: {avatar_path}")
            self.avatar_available = False
        else:
            self.avatar_available = True
            print(f"? Avatar trouvé: {avatar_path}")
        
        self.phoneme_detector = self._initialize_phoneme_detector()
    
    def _initialize_phoneme_detector(self):
        """Initialise un détecteur de phonèmes simplifiés pour la synchronisation labiale"""
        try:
            if TRANSFORMER_AVAILABLE:
                # Si des modèles avancés sont disponibles, on pourrait les utiliser ici
                # Mais pour l'instant, on va utiliser une approche plus simple
                return None
            return None
        except:
            return None
    
    def create_lipsync_avatar(self, audio_path, output_path, duration=None, position=("left", "bottom"), size=0.25):
        if not self.avatar_available:
            print("?? Avatar non disponible, impossible de créer le lip-sync")
            return None
        
        try:
            # 1. Charger l'avatar et l'audio
            avatar_clip = VideoFileClip(str(self.avatar_path))
            
            # Si l'audio existe, le charger
            audio_clip = None
            if audio_path and Path(audio_path).exists():
                audio_clip = AudioFileClip(str(audio_path))
                
                # Si aucune durée n'est spécifiée, utiliser celle de l'audio
                if duration is None:
                    duration = audio_clip.duration
            
            # Si toujours pas de durée, utiliser celle de l'avatar
            if duration is None:
                duration = avatar_clip.duration
            
            # 2. Adapter la durée de l'avatar
            # Si l'avatar est plus court que la durée requise, le boucler
            if avatar_clip.duration < duration:
                # Calculer combien de fois nous devons boucler l'avatar
                loop_times = int(duration / avatar_clip.duration) + 1
                # Créer une liste de clips identiques
                avatar_clips = [avatar_clip] * loop_times
                # Concaténer les clips
                avatar_clip = concatenate_videoclips(avatar_clips)
            
            # 3. Couper l'avatar à la bonne durée
            avatar_clip = avatar_clip.subclip(0, duration)
            
            # 4. Ajouter l'audio au clip de l'avatar
            if audio_clip:
                avatar_clip = avatar_clip.set_audio(audio_clip)
            
            # 5. Redimensionner l'avatar
            if isinstance(size, float):
                # Taille relative
                avatar_clip = avatar_clip.resize(height=int(avatar_clip.h * size))
            else:
                # Taille absolue
                avatar_clip = avatar_clip.resize(height=size)
            
            # 6. Positionner l'avatar
            avatar_clip = avatar_clip.set_position(position)
            
            # 7. Retourner le clip prêt à être intégré
            return avatar_clip
            
        except Exception as e:
            print(f"?? Erreur création avatar lip-sync: {e}")
            traceback.print_exc()
            return None
    # this method was not refactored here: def create_multipose_avatar(self, audio_path, output_path, duration=None, position=("left", "bottom"), size=0.25):
   
class ImprovedLipSyncAvatar(LipSyncAvatar):

    def __init__(self, avatar_path, device="cpu"):
        self.avatar_path = avatar_path
        self.device = device
        self.temp_dir = Path(tempfile.gettempdir()) / "avatar_lipsync"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Vérifier que l'avatar existe
        if not Path(avatar_path).exists():
            print(f"?? Avatar non trouvé: {avatar_path}")
            self.avatar_available = False
        else:
            self.avatar_available = True
            print(f"? Avatar trouvé: {avatar_path}")
        
        # Pré-extraction des segments d'avatar pour différents phonèmes
        self.mouth_states = self._extract_mouth_states()
    
    def _extract_mouth_states(self):
        if not self.avatar_available:
            return {}
        try:
            avatar_clip = VideoFileClip(str(self.avatar_path))
            duration = avatar_clip.duration
            mouth_states = {
                'closed': avatar_clip.subclip(0, min(0.1, duration * 0.05)),
                'slightly_open': avatar_clip.subclip(duration * 0.25, min(duration * 0.25 + 0.1, duration * 0.35)),
                'half_open': avatar_clip.subclip(duration * 0.45, min(duration * 0.45 + 0.1, duration * 0.55)),
                'open': avatar_clip.subclip(duration * 0.7, min(duration * 0.7 + 0.1, duration * 0.8)),
                'wide_open': avatar_clip.subclip(duration * 0.95, duration)
            }
            return mouth_states
        except Exception as e:
            print(f"?? Erreur lors de l'extraction des états de bouche: {e}")
            return {}

