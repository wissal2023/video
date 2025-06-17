# layout_manager.py
import re
import traceback
from PIL import Image, ImageStat, ImageOps, ImageEnhance
from moviepy.editor import ColorClip, TextClip, CompositeVideoClip, ImageClip

class SlideLayoutManager:
    """
    Gestionnaire de mise en page pour diapositives qui assure une séparation claire
    entre le texte et les images, évitant ainsi tout chevauchement.
    """
    
    def __init__(self, width=960, height=540, font_size_title=50, font_size_text=36):
        self.WIDTH = width
        self.HEIGHT = height
        self.FONT_SIZE_TITLE = font_size_title
        self.FONT_SIZE_TEXT = font_size_text
        self.TEXT_COLOR = "black"
        self.BG_COLOR = (255, 255, 255)
        self.TEXT_MARGIN = 40
        self.functions_replaced = False
    
    def apply_layout_fixes(self):
        global slide_clip_with_exact_markdown
        try:
            slide_clip_with_exact_markdown = self.create_slide_with_separated_content
            self.replace_layout_optimizer()
            self.functions_replaced = True
            print("? Mise en page des diapositives optimisée pour éviter le chevauchement texte/image")
            return True
        except Exception as e:
            print(f"? Erreur lors de l'application des corrections de mise en page: {e}")
            traceback.print_exc()
            return False
    
    def create_slide_with_separated_content(self, title, content, images, style_params=None, layout_params=None, duration=8):
        """
        Crée un slide qui préserve exactement la structure Markdown originale
        Traite tous les éléments visuels (images, tableaux, graphiques) sans texte en mode plein écran
        """        
        # Déterminer si le contenu ne contient que des éléments visuels (images, tableaux) sans texte
        has_table = re.search(r'\|[-]+\|', content) is not None
        has_table = has_table or "|" in content and any(line.count('|') > 2 for line in content.split('\n'))
        
        # Supprimer les références d'images et les tableaux pour voir s'il reste du texte
        clean_content = re.sub(r'!\[.*?\]\(.*?\)', '', content).strip()
        if has_table:
            clean_content_without_tables = re.sub(r'^\|.*\|$', '', clean_content, flags=re.MULTILINE)
            clean_content_without_tables = re.sub(r'^[-|:]+$', '', clean_content_without_tables, flags=re.MULTILINE)
            clean_content = clean_content_without_tables
        
        # Supprimer tous les espaces et caractères de formatage
        clean_content = re.sub(r'[#*_\s\n\r]', '', clean_content)
        
        # Définir si c'est un contenu visuel sans texte (image ou tableau)
        is_visual_only = not bool(clean_content) and (bool(images) or has_table)
        
        # Valeurs par défaut pour les paramètres de style et de mise en page
        if style_params is None:
            style_params = {
                'name': 'balanced',
                'text_size': self.FONT_SIZE_TEXT,
                'animation_level': 0.5,
                'color_scheme': 'neutral',
                'layout': 'balanced',
                'transition': 'fade'
            }
        
        if layout_params is None:
            layout_params = {}
        
        # Définir explicitement le mode plein écran si c'est un contenu visuel sans texte
        layout_params['fullscreen_image'] = is_visual_only
        
        # Créer l'arrière-plan
        bg_color = self.BG_COLOR
        if style_params.get('color_scheme') == 'vibrant':
            bg_color = (240, 250, 255)  # Bleu très pâle
        elif style_params.get('color_scheme') == 'monochrome':
            bg_color = (248, 248, 248)  # Gris clair
        elif style_params.get('color_scheme') == 'contrast':
            bg_color = (235, 235, 245)  # Violet très pâle
        elif style_params.get('color_scheme') == 'focused':
            bg_color = (245, 245, 245)  # Gris très pâle
        
        bg = ColorClip((self.WIDTH, self.HEIGHT), color=bg_color).set_duration(duration)
        layers = [bg]
        
        # Si c'est un mode contenu visuel plein écran (image ou tableau sans texte), traiter différemment
        if is_visual_only:
            print(f"?? Mode plein écran activé - contenu visuel uniquement")
            
            # Ajouter le titre s'il existe (en bas pour ne pas gêner le contenu visuel)
            if title:
                title_clip = TextClip(
                    title, 
                    fontsize=self.FONT_SIZE_TITLE, 
                    color=self.TEXT_COLOR,
                    method="caption", 
                    align="center",
                    size=(self.WIDTH-120, None)
                ).set_position(("center", self.HEIGHT - 60)).set_duration(duration)
                
                layers.append(title_clip)
            
            # Si c'est un tableau (pas d'images), on le traite différemment
            if has_table and not images:
                # Pour les tableaux sans images, on applique un style de mise en page adapté
                # qui affiche le tableau en grand, centralisé
                table_content = content
                
                # Créer un clip de texte pour le tableau qui occupe presque tout l'écran
                table_clip = TextClip(
                    table_content, 
                    fontsize=self.FONT_SIZE_TABLE, 
                    color=self.TEXT_COLOR,
                    method="caption", 
                    align="center",
                    size=(self.WIDTH * 0.9, None)
                ).set_position(("center", self.HEIGHT * 0.1)).set_duration(duration)
                
                layers.append(table_clip)
                
            # Si des images sont présentes, les afficher en plein écran
            elif images:
                try:
                    for img_path in images:
                        try:
                            img_clip = ImageClip(str(img_path))
                            
                            # Déterminer la meilleure taille pour couvrir l'écran tout en préservant le ratio
                            img_ratio = img_clip.h / img_clip.w
                            screen_ratio = self.HEIGHT / self.WIDTH
                            
                            if img_ratio > screen_ratio:  # Image plus haute que large
                                new_width = self.WIDTH * 0.9  # 90% de la largeur
                                new_height = new_width * img_ratio
                            else:  # Image plus large que haute
                                new_height = self.HEIGHT * 0.8  # 80% de la hauteur (laisse place au titre)
                                new_width = new_height / img_ratio
                            
                            # Redimensionner l'image
                            img_clip = img_clip.resize(width=new_width) if img_ratio > screen_ratio else img_clip.resize(height=new_height)
                            
                            # Centrer l'image
                            x_pos = (self.WIDTH - img_clip.w) / 2
                            y_pos = (self.HEIGHT - img_clip.h) / 2 - 20  # Légèrement plus haut pour laisser de l'espace au titre
                            
                            # Appliquer des animations si style animé
                            animation_level = style_params.get('animation_level', 0.5)
                            if animation_level > 0.6:
                                img_clip = img_clip.crossfadein(min(1.0, animation_level))
                            
                            img_clip = img_clip.set_position((x_pos, y_pos)).set_duration(duration)
                            layers.append(img_clip)
                            break  # N'utiliser que la première image
                        except Exception as e:
                            print(f"?? Erreur chargement image {img_path}: {e}")
                except Exception as e:
                    print(f"?? Erreur générale images: {e}")
        
        else:
            # Affichage standard pour le contenu avec texte
            # Réserver de l'espace pour le titre
            title_height = 60
            
            # Convertir les images en liste de chemins si c'est une liste de dictionnaires
            image_paths = []
            if images:
                for img in images:
                    if isinstance(img, dict):
                        image_paths.append(img.get('path', ''))
                    elif isinstance(img, tuple):  # Support de l'ancien format
                        image_paths.append(img[0])
                    else:
                        image_paths.append(img)
            
            # Déterminer si nous avons des images valides
            has_valid_images = bool(image_paths)
            
            # Ajouter le titre s'il existe
            if title:
                # Vérifier si le titre est un marqueur de page pour le masquer
                if not re.match(r'(---\s*)?[Pp]age\s+\d+(\s*---)?', title):
                    title_position = layout_params.get('title_position', 'top')
                    title_y = self.HEIGHT - 50 if title_position == 'bottom' else 10
                    
                    title_clip = TextClip(
                        title, 
                        fontsize=self.FONT_SIZE_TITLE, 
                        color=self.TEXT_COLOR,
                        method="caption", 
                        align="center",
                        size=(self.WIDTH-120, None)
                    ).set_position(("center", title_y)).set_duration(duration)
                    
                    layers.append(title_clip)
            
            # Appliquer les valeurs spécifiques des layout_params pour le texte
            text_x = int(layout_params.get('text_x', 0.1) * self.WIDTH)
            text_width = int(layout_params.get('text_width', 0.8) * self.WIDTH)
            text_y = int(layout_params.get('text_y', 0.15) * self.HEIGHT)
            
            # Supprimer les références d'images du texte affiché
            processed_content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
            
            # Préserver le formatage Markdown (gras, italique, listes)
            processed_content = self.preserve_markdown_formatting(processed_content)
            
            # Vérifier si le texte a du contenu après nettoyage
            if processed_content.strip():
                # Créer le clip de texte avec l'animation selon le niveau d'animation
                text_size = int(style_params.get('text_size', self.FONT_SIZE_TEXT) * layout_params.get('text_size_factor', 1.0))
                animation_level = style_params.get('animation_level', 0.5)
                
                if animation_level > 0.3:
                    # Animation plus élaborée pour les styles dynamiques
                    text_clip = self.create_animated_text_clip(
                        processed_content, 
                        duration=duration,
                        animation_level=animation_level,
                        text_size=text_size,
                        text_width=text_width
                    )
                else:
                    # Texte statique pour les styles minimalistes
                    text_clip = self.create_text_clip(
                        processed_content, 
                        fontsize=text_size, 
                        width=text_width,
                        position=(text_x, text_y), 
                        duration=duration
                    )
                
                # Ajouter le texte à la diapositive
                layers.append(text_clip)
            
            # Ajouter les images si présentes avec positionnement spécifique
            if has_valid_images:
                try:
                    # Récupérer les paramètres de mise en page pour l'image
                    img_x = int(layout_params.get('image_x', 0.55) * self.WIDTH)
                    img_y = int(layout_params.get('image_y', 0.15) * self.HEIGHT)
                    img_width = int(layout_params.get('image_width', 0.4) * self.WIDTH)
                    
                    # Utiliser la première image valide
                    for img_path in image_paths:
                        try:
                            img_clip = ImageClip(str(img_path))
                            img_ratio = img_clip.h / img_clip.w
                            img_height = int(img_width * img_ratio)
                            
                            # Limiter la hauteur si nécessaire
                            max_height = int(self.HEIGHT * 0.6)
                            if img_height > max_height:
                                img_height = max_height
                                img_width = int(img_height / img_ratio)
                            
                            # Redimensionner et positionner l'image
                            img_clip = img_clip.resize(width=img_width)
                            img_clip = img_clip.set_position((img_x, img_y)).set_duration(duration)
                            
                            # Ajouter un effet de fondu d'entrée pour les styles dynamiques
                            animation_level = style_params.get('animation_level', 0.5)
                            if animation_level > 0.6:
                                img_clip = img_clip.crossfadein(min(1.0, animation_level))
                            
                            layers.append(img_clip)
                            break  # Utiliser seulement la première image valide
                        except Exception as e:
                            print(f"?? Erreur chargement image {img_path}: {e}")
                            continue
                except Exception as e:
                    print(f"?? Erreur générale images: {e}")
        
        # Ajouter le logo si une fonction handle_logo existe
        if 'handle_logo' in globals():
            layers = handle_logo(layers, style_params, duration)
        
        # Créer la diapositive finale
        return CompositeVideoClip(layers, size=(self.WIDTH, self.HEIGHT))
  
    def replace_layout_optimizer(self):
        if 'LayoutOptimizer' in globals():
            LayoutOptimizer.optimize_layout = self.compute_optimal_layout
            print("? Fonction d'optimisation de mise en page remplacée")
        else:
            print("?? Classe LayoutOptimizer non trouvée dans le contexte global")
    
    def compute_optimal_layout(self, content, title, images, style_params):
        # Implementation of layout optimization logic
        pass  # To be implemented or imported from existing code
