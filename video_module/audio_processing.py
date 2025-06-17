import pyttsx3
import shutil
import traceback
from pathlib import Path
import torch
import numpy as np
import librosa
import soundfile as sf
import tempfile
import re
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from pydub import AudioSegment, effects
import time
from moviepy.editor import AudioFileClip, concatenate_audioclips

class EnhancedVoiceSynthesizer:
    """
    Classe avancée pour la synthèse vocale avec des capacités de synchronisation
    et de traitement audio optimisés pour les présentations
    """
    
    def __init__(self, voice_rate=150, voice_volume=1.0):
        # Initialiser le moteur de synthèse vocale
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', voice_rate)    # Vitesse de parole
        self.engine.setProperty('volume', voice_volume)  # Volume (0 à 1)
        
        # Analyser les voix disponibles pour en choisir une de qualité
        self.selected_voice = None
        self.initialize_voice()
        
        # Paramètres pour une meilleure qualité audio
        self.temp_dir = Path(tempfile.gettempdir()) / "slide_audio"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Modèle d'analyse prosodique pour améliorer les pauses et l'accent
        self.prosody_model = self.initialize_prosody_model()
        
        # Formats de fichiers supportés
        self.supported_formats = ['wav', 'mp3', 'aac', 'ogg']
        
        # Chemins de caches pour éviter de regénérer les mêmes audios
        self.audio_cache = {}
        
        print("? Synthétiseur vocal amélioré initialisé")
    
    def initialize_voice(self):
        """Sélectionne la meilleure voix disponible pour la synthèse"""
        try:
            voices = self.engine.getProperty('voices')
            if not voices:
                print("?? Aucune voix trouvée, utilisation des paramètres par défaut")
                return
            
            # Priorités: Voix française de bonne qualité > Voix féminine > Autre
            french_voices = []
            female_voices = []
            other_voices = []
            
            for voice in voices:
                # Vérifier si la voix est française
                is_french = False
                if hasattr(voice, 'languages'):
                    is_french = any('fr' in str(lang).lower() for lang in voice.languages)
                
                # Vérifier si la voix est féminine
                is_female = False
                if hasattr(voice, 'gender'):
                    is_female = voice.gender.lower() == 'female'
                elif hasattr(voice, 'name') and any(term in voice.name.lower() for term in ['female', 'femme', 'woman']):
                    is_female = True
                
                # Ajouter à la liste appropriée
                if is_french:
                    french_voices.append(voice)
                elif is_female:
                    female_voices.append(voice)
                else:
                    other_voices.append(voice)
            
            # Sélectionner selon les priorités
            if french_voices:
                self.selected_voice = french_voices[0]
                print(f"? Voix française sélectionnée: {self.selected_voice.name}")
            elif female_voices:
                self.selected_voice = female_voices[0]
                print(f"? Voix féminine sélectionnée: {self.selected_voice.name}")
            elif other_voices:
                self.selected_voice = other_voices[0]
                print(f"? Voix standard sélectionnée: {self.selected_voice.name}")
            
            # Configurer la voix sélectionnée
            if self.selected_voice:
                self.engine.setProperty('voice', self.selected_voice.id)
        
        except Exception as e:
            print(f"?? Erreur sélection voix: {e}")
    
    def initialize_prosody_model(self):
        """Initialise un modèle simple pour analyser et améliorer la prosodie"""
        model = {
            'punctuation_pauses': {
                '.': 0.5,    # Pause longue pour point
                ',': 0.3,    # Pause moyenne pour virgule
                ';': 0.4,    # Pause moyenne-longue pour point-virgule
                ':': 0.4,    # Pause moyenne-longue pour deux-points
                '?': 0.5,    # Pause longue pour interrogation
                '!': 0.5,    # Pause longue pour exclamation
                '-': 0.3,    # Pause moyenne pour tiret
                '(': 0.2,    # Légère pause pour parenthèse ouvrante
                ')': 0.2,    # Légère pause pour parenthèse fermante
            },
            'structural_pauses': {
                '\n\n': 0.7,  # Pause pour nouvelle section
                '* ': 0.4,    # Pause pour puce
                '- ': 0.4,    # Pause pour tiret liste
                '# ': 0.8,    # Pause pour titre
            },
            'emphasis_patterns': [
                r'\*\*(.*?)\*\*',  # Texte en gras
                r'__(.*?)__',      # Souligné
                r'!important',      # Marqueur d'importance
            ]
        }
        return model
    
    def optimize_narration_text(self, text):
        """
        Optimise le texte pour une meilleure narration en ajoutant
        des marqueurs SSML ou des pauses appropriées
        """
        if not text:
            return text
        
        try:
            # Nettoyer le texte du formatage Markdown
            clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Supprimer les références d'images
            clean_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_text)  # Remplacer les liens par leur texte
            
            # Remplacer les styles Markdown par du texte brut
            clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)  # Gras
            clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)      # Italique
            
            # Améliorer la structure pour les listes
            clean_text = re.sub(r'^-\s+', '• ', clean_text, flags=re.MULTILINE)  # Remplacer - par •
            
            # Diviser les phrases trop longues
            sentences = re.split(r'([.!?])', clean_text)
            optimized_sentences = []
            
            i = 0
            while i < len(sentences) - 1:
                sentence = sentences[i]
                punctuation = sentences[i+1] if i+1 < len(sentences) else ""
                
                # Diviser les phrases trop longues
                if len(sentence.split()) > 25:
                    words = sentence.split()
                    mid = len(words) // 2
                    
                    # Chercher un point de coupure naturel
                    for j in range(mid-3, mid+3):
                        if 0 <= j < len(words) and any(p in words[j] for p in [',', ';', ':', '-']):
                            mid = j + 1
                            break
                    
                    first_part = ' '.join(words[:mid])
                    second_part = ' '.join(words[mid:])
                    
                    # Capitaliser la première lettre de la seconde partie
                    if second_part:
                        second_part = second_part[0].upper() + second_part[1:] if len(second_part) > 1 else second_part.upper()
                    
                    optimized_sentences.append(first_part + ".")
                    optimized_sentences.append(second_part + punctuation)
                else:
                    optimized_sentences.append(sentence + punctuation)
                
                i += 2
            
            # Ajouter le reste des segments
            if i < len(sentences):
                optimized_sentences.append(sentences[i])
            
            # Recombiner avec des pauses
            optimized_text = ' '.join(optimized_sentences)
            
            # Ajouter des pauses à des endroits stratégiques
            for punctuation, pause_duration in self.prosody_model['punctuation_pauses'].items():
                if punctuation in ['.', '?', '!']:
                    # Pour les ponctuations finales, ajouter une pause plus longue
                    optimized_text = optimized_text.replace(f"{punctuation} ", f"{punctuation} <break time='{int(pause_duration * 1000)}ms'/> ")
                else:
                    # Pour les autres ponctuations, ajouter une pause plus courte
                    optimized_text = optimized_text.replace(f"{punctuation} ", f"{punctuation} <break time='{int(pause_duration * 1000)}ms'/> ")
            
            # Normaliser les nombres et abréviations pour une meilleure prononciation
            optimized_text = re.sub(r'(\d+)%', r'\1 pour cent', optimized_text)
            optimized_text = re.sub(r'Dr\.', 'Docteur', optimized_text)
            optimized_text = re.sub(r'M\.', 'Monsieur', optimized_text)
            optimized_text = re.sub(r'Mme\.', 'Madame', optimized_text)
            
            # Améliorer la prononciation des sigles
            # Ajouter des espaces entre les lettres pour une meilleure prononciation
            optimized_text = re.sub(r'\b([A-Z]{2,})\b', lambda m: ' '.join(list(m.group(1))), optimized_text)
            
            # Remplacer les marqueurs SSML par des espaces pour la synthèse
            optimized_text = re.sub(r'<break.*?/>', ' ', optimized_text)
            
            return optimized_text
        
        except Exception as e:
            print(f"?? Erreur optimisation texte narration: {e}")
            return text  # Retourner le texte original en cas d'erreur
    
    def say_text(self, text):
        """Prononce le texte immédiatement avec les optimisations"""
        try:
            # Optimiser le texte pour la narration
            optimized_text = self.optimize_narration_text(text)
            
            # Prononcer le texte
            self.engine.say(optimized_text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"?? Erreur synthèse vocale: {e}")
            return False
    
    def save_to_file(self, text, output_path, format='wav'):
        """
        Sauvegarde le texte narré en tant que fichier audio avec optimisations
        
        Args:
            text: Texte à prononcer
            output_path: Chemin du fichier de sortie
            format: Format audio (wav, mp3, etc.)
            
        Returns:
            bool: True si réussi, False sinon
        """
        try:
            # Vérifier si on a déjà généré cet audio (cache)
            cache_key = f"{text}_{Path(output_path).name}"
            if cache_key in self.audio_cache and Path(self.audio_cache[cache_key]).exists():
                # Copier le fichier depuis le cache
                shutil.copy(self.audio_cache[cache_key], output_path)
                print(f"? Audio récupéré du cache: {output_path}")
                return True
            
            # Optimiser le texte pour la narration
            optimized_text = self.optimize_narration_text(text)
            
            # Vérifier le format demandé
            if format.lower() not in self.supported_formats:
                print(f"?? Format {format} non supporté, utilisation de wav")
                format = 'wav'
            
            # Créer un fichier temporaire pour la synthèse
            temp_wav = self.temp_dir / f"temp_{time.time()}.wav"
            
            # Sauvegarder l'audio brut
            self.engine.save_to_file(optimized_text, str(temp_wav))
            self.engine.runAndWait()
            
            # Vérifier que le fichier a été créé
            if not temp_wav.exists() or temp_wav.stat().st_size == 0:
                print(f"?? Échec de génération du fichier audio temporaire")
                return False
            
            # Améliorer la qualité audio (normalisation, débruitage, etc.)
            enhanced_audio = self.enhance_audio_quality(temp_wav)
            
            # Convertir si nécessaire
            if format.lower() == 'wav':
                # Copier directement
                shutil.copy(enhanced_audio, output_path)
            else:
                # Convertir vers le format demandé
                self.convert_audio_format(enhanced_audio, output_path, format)
            
            # Nettoyer les fichiers temporaires
            try:
                if temp_wav.exists():
                    temp_wav.unlink()
                if enhanced_audio != temp_wav and enhanced_audio.exists():
                    enhanced_audio.unlink()
            except:
                pass
            
            # Vérifier le fichier final
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                # Ajouter au cache
                self.audio_cache[cache_key] = str(output_path)
                print(f"? Audio sauvegardé dans {output_path}")
                return True
            else:
                print(f"?? Fichier audio final non créé ou vide")
                return False
                
        except Exception as e:
            print(f"?? Erreur lors de la sauvegarde audio: {e}")
            traceback.print_exc()
            return False
    
    def enhance_audio_quality(self, audio_path):
        """
        Améliore la qualité d'un fichier audio (normalisation, suppression de silence, etc.)
        
        Args:
            audio_path: Chemin du fichier audio à améliorer
            
        Returns:
            Path: Chemin du fichier audio amélioré
        """
        try:
            # Essayer d'utiliser pydub pour l'amélioration audio
            try:
                # Charger l'audio
                audio = AudioSegment.from_wav(str(audio_path))
                
                # Normalisation du volume
                audio = normalize(audio)
                
                # Supprimer les silences trop longs au début et à la fin
                silence_threshold = -50  # dB
                min_silence_len = 700    # ms
                
                # Détecter le début (ignorer le silence initial)
                def detect_leading_silence(sound, silence_threshold, chunk_size=10):
                    trim_ms = 0
                    while trim_ms < len(sound) and sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
                        trim_ms += chunk_size
                    return trim_ms
                
                start_trim = detect_leading_silence(audio, silence_threshold)
                end_trim = detect_leading_silence(audio.reverse(), silence_threshold)
                
                # Découper l'audio
                duration = len(audio)
                trimmed_audio = audio[start_trim:duration-end_trim]
                
                # Sauvegarder l'audio amélioré
                enhanced_path = audio_path.with_suffix('.enhanced.wav')
                trimmed_audio.export(enhanced_path, format="wav")
                
                return enhanced_path
                
            except ImportError:
                print("?? pydub non disponible, utilisation de l'audio original")
                return audio_path
                
        except Exception as e:
            print(f"?? Erreur amélioration audio: {e}")
            return audio_path  # Retourner le chemin original en cas d'erreur
    
    def convert_audio_format(self, input_path, output_path, target_format):
        """
        Convertit un fichier audio vers un autre format
        
        Args:
            input_path: Chemin du fichier audio source
            output_path: Chemin du fichier audio de destination
            target_format: Format cible (mp3, wav, etc.)
            
        Returns:
            bool: True si réussi, False sinon
        """
        try:
            # Essayer d'utiliser pydub pour la conversion
            try:                
                # Charger l'audio source
                audio = AudioSegment.from_file(str(input_path))
                
                # Exporter dans le format cible
                audio.export(output_path, format=target_format)
                
                return True
                
            except ImportError:
                print("?? pydub non disponible, tentative de copie directe")
                shutil.copy(input_path, output_path)
                return True
                
        except Exception as e:
            print(f"?? Erreur conversion audio: {e}")
            return False
    
    def describe_image_with_voice(self, image_path, text_processor, output_path=None):
        """
        Génère et prononce une description d'image
        
        Args:
            image_path: Chemin de l'image
            text_processor: Processeur de texte pour la description
            output_path: Chemin du fichier audio de sortie (optionnel)
            
        Returns:
            bool: True si réussi, False sinon
        """
        try:
            # Obtenir la description de l'image
            description = text_processor.describe_image(image_path, detailed=True)
            
            if not description:
                print(f"?? Pas de description générée pour {image_path}")
                return False
            
            print(f"??? Description générée: {description}")
            
            # Ajouter une phrase d'introduction
            description = f"La diapositive contient une image qui montre {description}"
            
            # Si un chemin de sortie est fourni, enregistrer l'audio
            if output_path:
                return self.save_to_file(description, output_path)
            else:
                # Sinon, prononcer la description directement
                return self.say_text(description)
                
        except Exception as e:
            print(f"?? Erreur description vocale image: {e}")
            return False
    
    def create_combined_audio(self, audio_paths, output_path, crossfade_duration=1000):
        """
        Combine plusieurs fichiers audio en un seul avec transitions fluides
        
        Args:
            audio_paths: Liste des chemins audio à combiner
            output_path: Chemin du fichier audio combiné
            crossfade_duration: Durée du fondu enchaîné en ms
            
        Returns:
            bool: True si réussi, False sinon
        """
        if not audio_paths:
            print("?? Aucun fichier audio à combiner")
            return False
        
        # Si un seul fichier, le copier directement
        if len(audio_paths) == 1 and Path(audio_paths[0]).exists():
            try:
                shutil.copy(audio_paths[0], output_path)
                return True
            except Exception as e:
                print(f"?? Erreur copie audio: {e}")
                return False
        
        try:
            # Essayer d'utiliser pydub pour la combinaison
            try:                
                # Charger le premier fichier
                combined = AudioSegment.from_file(str(audio_paths[0]))
                
                # Ajouter les fichiers suivants avec crossfade
                for audio_path in audio_paths[1:]:
                    if not Path(audio_path).exists():
                        print(f"?? Fichier audio non trouvé: {audio_path}")
                        continue
                    
                    next_segment = AudioSegment.from_file(str(audio_path))
                    
                    # Ajouter un silence entre les segments
                    silence = AudioSegment.silent(duration=200)  # 200ms
                    combined += silence
                    
                    # Si les deux segments sont assez longs, faire un crossfade
                    if len(combined) > crossfade_duration and len(next_segment) > crossfade_duration:
                        combined = combined.append(next_segment, crossfade=crossfade_duration)
                    else:
                        combined += next_segment
                
                # Normaliser le volume final
                combined = normalize(combined)
                
                # Exporter le fichier combiné
                combined.export(output_path, format="wav")
                
                print(f"? {len(audio_paths)} fichiers audio combinés dans {output_path}")
                return True
                
            except ImportError:
                # Solution alternative avec moviepy
                try:
                    # Charger tous les clips audio valides
                    audio_clips = []
                    for audio_path in audio_paths:
                        if Path(audio_path).exists():
                            clip = AudioFileClip(str(audio_path))
                            audio_clips.append(clip)
                    
                    if not audio_clips:
                        print("?? Aucun clip audio valide trouvé")
                        return False
                    
                    # Concaténer les clips
                    final_clip = concatenate_audioclips(audio_clips)
                    
                    # Sauvegarder le résultat
                    final_clip.write_audiofile(str(output_path))
                    
                    # Fermer les clips
                    for clip in audio_clips:
                        clip.close()
                    final_clip.close()
                    
                    print(f"? {len(audio_clips)} clips audio combinés dans {output_path}")
                    return True
                    
                except Exception as e:
                    print(f"?? Erreur combinaison audio avec moviepy: {e}")
                    return False
                
        except Exception as e:
            print(f"?? Erreur combinaison audio: {e}")
            return False
