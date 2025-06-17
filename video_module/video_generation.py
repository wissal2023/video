# video_generation.py
import os
import re
import time
import json
import random
import traceback
import shutil
import tempfile
import torch
import pyttsx3
import numpy as np
from pathlib import Path
from moviepy.editor import TextClip, ImageClip, ColorClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip, concatenate_audioclips
from video_module.text_processing import TextProcessor
from video_module.audio_processing import EnhancedVoiceSynthesizer
from video_module.slide_models import SlideQualityModel, SlideStyleRecommender, LayoutOptimizer
from video_module.content_segmentation import ContentSegmenter

class MLEnhancedVideoGenerator:
    """
    this is the original init method
    """    
    def __init__(self, base_dir=Path(__file__).parent, output_dir=None, model_path=None):
        self.base_dir = base_dir
        self.output_dir = output_dir or base_dir / "output"
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir = base_dir / "models"
        self.model_dir.mkdir(exist_ok=True)
        
        # Constantes de base
        self.FONT_SIZE_TITLE = 50
        self.FONT_SIZE_TEXT = 36
        self.FONT_SIZE_TABLE = 28
        self.WIDTH, self.HEIGHT = 960, 540
        self.BG_COLOR = (255, 255, 255)
        self.TEXT_COLOR = "black"
        self.TEXT_MARGIN = 40
        self.SLIDE_DUR = 8
        self.TRANSITION_DUR = 1
        
        # Initialiser le système de feedback
        #self.feedback_system = UserFeedbackSystem()
        
        # Charger ou initialiser les modèles ML
        self.slide_quality = SlideQualityModel()
        self.content_segmenter = ContentSegmenter()
        self.style_recommender = SlideStyleRecommender()
        self.layout_optimizer = LayoutOptimizer()
        #self.readability_evaluator = ReadabilityEvaluator()
        
        # Historique d'apprentissage
        self.learning_history = []
        
        # Synthèse vocale
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)      
        # Initialiser le processeur de texte avec le modèle spécifié ou le modèle par défaut
        self.text_processor = TextProcessor(model_path=model_path)
        self.voice_synthesizer = EnhancedVoiceSynthesizer()
        
        # Dossier pour les fichiers audio des descriptions d'images
        self.image_descriptions_dir = output_dir / "image_descriptions"
        self.image_descriptions_dir.mkdir(exist_ok=True)        
        print("? Générateur de vidéo ML initialisé")
             
    def process_markdown(self, markdown_content):
        """
        Traite un fichier Markdown et génère une présentation vidéo optimisée
        avec l'apprentissage automatique
        
        Args:
            markdown_content: Contenu Markdown à transformer
            
        Returns:
            output_path: Chemin du fichier vidéo généré
        """
        try:
            # 1. Extraire images et texte
            # For simplicity, assume images are extracted elsewhere or none
            
            # 2. Segmenter le contenu en sections optimales pour des diapositives
            segments = self.content_segmenter.segment_content(markdown_content)
            
            slides = []
            slide_data = []
            
            for i, segment in enumerate(segments):
                title = segment.get('title', f"Section {i+1}")
                content = segment.get('content', "")
                has_image = segment.get('has_image', False)
                
                # Recommander le style optimal
                style_params = self.style_recommender.recommend_style(content, title, has_image)
                if not isinstance(style_params, dict):
                    style_params = {
                        'name': 'balanced',
                        'text_size': self.FONT_SIZE_TEXT,
                        'animation_level': 0.5,
                        'color_scheme': 'neutral',
                        'layout': 'balanced',
                        'transition': 'fade'
                    }
                if 'transition' not in style_params:
                    style_params['transition'] = 'fade'
                
                # Optimiser la mise en page
                layout_params = self.layout_optimizer.optimize_layout(content, title, [], style_params)
                
                # Créer la diapositive (simplified)
                slide = self.create_slide(title, content, style_params, layout_params)
                
                slides.append({
                    'slide': slide,
                    'title': title,
                    'style': style_params.get('name', 'balanced'),
                    'transition': style_params.get('transition', 'fade'),
                    'duration': self.SLIDE_DUR
                })
                
                slide_data.append({
                    'title': title,
                    'content': content,
                    'has_image': has_image,
                    'style': style_params.get('name', 'balanced')
                })
            
            if not slides:
                print("? Aucune diapositive n'a pu être créée")
                return None
            
            # Générer la narration audio
            narration_audio_paths = []
            for i, segment in enumerate(segments):
                narration_text = self.voice_synthesizer.optimize_narration_text(segment.get('content', ''))
                if narration_text and narration_text.strip():
                    audio_filename = f"narration_{i+1}.wav"
                    audio_path = self.output_dir / audio_filename
                    success = self.voice_synthesizer.save_to_file(narration_text, audio_path)
                    if success and audio_path.exists():
                        narration_audio_paths.append(audio_path)
            
            # Assembler les clips avec transitions
            final_clips = []
            for i, slide_info in enumerate(slides):
                final_clips.append(slide_info['slide'])
            
            final_video = concatenate_videoclips(final_clips)
            
            # Ajouter l'audio combiné si disponible
            if narration_audio_paths:
                combined_audio_path = self.output_dir / "combined_narration.wav"
                self.voice_synthesizer.create_combined_audio(narration_audio_paths, combined_audio_path)
                if combined_audio_path.exists():
                    narration_audio = AudioFileClip(str(combined_audio_path))
                    final_video = final_video.set_audio(narration_audio)
            
            output_path = self.output_dir / "ml_presentation.mp4"
            final_video.write_videofile(str(output_path), fps=24, codec='libx264', audio_codec='aac')
            
            print(f"? Présentation sauvegardée: {output_path}")
            return output_path
        
        except Exception as e:
            print(f"? Erreur lors du traitement: {e}")
            traceback.print_exc()
            return None
    
    def create_slide(self, title, content, style_params, layout_params):
        """
        Crée une diapositive simple avec texte (simplified)
        """
        bg = ColorClip((self.WIDTH, self.HEIGHT), color=self.BG_COLOR).set_duration(self.SLIDE_DUR)
        layers = [bg]
        
        if title:
            title_clip = TextClip(title, fontsize=self.FONT_SIZE_TITLE, color=self.TEXT_COLOR, method="caption", size=(self.WIDTH-120, None))
            title_clip = title_clip.set_position(("center", 10)).set_duration(self.SLIDE_DUR)
            layers.append(title_clip)
        
        processed_content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        if processed_content.strip():
            text_clip = TextClip(processed_content, fontsize=style_params.get('text_size', self.FONT_SIZE_TEXT), color=self.TEXT_COLOR, method="caption", size=(int(self.WIDTH*0.8), None))
            text_clip = text_clip.set_position((int(self.WIDTH*0.1), int(self.HEIGHT*0.15))).set_duration(self.SLIDE_DUR)
            layers.append(text_clip)
        
        slide = CompositeVideoClip(layers, size=(self.WIDTH, self.HEIGHT))
        return slide
