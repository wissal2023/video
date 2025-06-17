"""
create_clean_slides.py - Crée des diapositives propres dans le style HTML pour toutes vos vidéos
"""

import os
import re
import time
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
import fitz  # PyMuPDF
from pathlib import Path
            
def create_directory(path):
    """Crée un répertoire s'il n'existe pas"""
    os.makedirs(path, exist_ok=True)
    return path

def get_available_fonts():
    """Retourne une liste des polices disponibles"""
    try:
        import matplotlib.font_manager as fm
        font_paths = fm.findSystemFonts()
        fonts = []
        
        for font_path in font_paths:
            try:
                font = fm.FontProperties(fname=font_path)
                font_name = font.get_name()
                if "segoe" in font_name.lower() or "arial" in font_name.lower():
                    fonts.append((font_name, font_path))
            except:
                pass
        
        return fonts
    except:
        return []

def create_html_style_slide(title, content, image_path=None, output_path=None, slide_num=0):
    """
    Crée une diapositive dans le style exact de la page HTML
    
    Args:
        title: Titre de la diapositive
        content: Contenu textuel
        image_path: Chemin de l'image à inclure (optionnel)
        output_path: Chemin où sauvegarder l'image
        slide_num: Numéro de la diapositive
    
    Returns:
        Le chemin de l'image créée
    """
    # Dimensions de l'image (comme dans la page HTML)
    width, height = 1200, 800
    
    # Styles CSS convertis pour PIL
    styles = {
        'bg_color': '#FFFFFF',  # Blanc
        'title_color': '#2c3e50',  # Bleu foncé pour h1
        'subtitle_color': '#3498db',  # Bleu pour h2
        'text_color': '#333333',  # Gris foncé pour le texte
        'border_color': '#dddddd',  # Gris clair pour les bordures
        'shadow_color': 'rgba(0, 0, 0, 0.1)',  # Noir avec 10% d'opacité
        'highlight_bg': '#f8f9fa',  # Fond pour la section de surbrillance
        'highlight_border': '#3498db',  # Bordure de la section de surbrillance
        'image_container_bg': '#f9f9f9',  # Fond pour le conteneur d'image
    }
    
    # Créer une nouvelle image avec fond blanc
    img = Image.new('RGB', (width, height), color=styles['bg_color'])
    draw = ImageDraw.Draw(img)
    
    # Chercher des polices adaptées
    available_fonts = get_available_fonts()
    title_font_path = None
    text_font_path = None
    
    for font_name, font_path in available_fonts:
        if "segoe ui bold" in font_name.lower() or "arial bold" in font_name.lower():
            title_font_path = font_path
        elif "segoe ui" in font_name.lower() or "arial" in font_name.lower():
            text_font_path = font_path
    
    # Polices par défaut si les polices spécifiques ne sont pas trouvées
    if not title_font_path:
        title_font = ImageFont.load_default()
        title_font_size = 36
    else:
        title_font = ImageFont.truetype(title_font_path, 36)
        title_font_size = 36
    
    if not text_font_path:
        text_font = ImageFont.load_default()
        text_font_size = 24
    else:
        text_font = ImageFont.truetype(text_font_path, 24)
        text_font_size = 24
    
    # Dessiner le titre principal
    title_y = 20
    draw.text((20, title_y), title, font=title_font, fill=styles['title_color'])
    title_height = 50  # Hauteur approximative
    
    # Créer une section de contenu
    margin = 20
    content_y = title_y + title_height + margin
    
    # Définir les dimensions et positions selon qu'il y a une image ou non
    if image_path and os.path.exists(image_path):
        # Mise en page avec texte à gauche, image à droite
        has_image = True
        content_width = (width - 3*margin) // 2
        image_x = width // 2 + margin // 2
        image_width = (width - 3*margin) // 2
    else:
        # Mise en page avec texte sur toute la largeur
        has_image = False
        content_width = width - 2*margin
    
    # Créer un cadre pour le contenu
    content_height = height - content_y - margin
    content_x = margin
    
    # Dessiner la bordure du cadre de contenu
    border_radius = 8
    draw.rounded_rectangle(
        [(content_x, content_y), (content_x + content_width, content_y + content_height)],
        radius=border_radius,
        outline=styles['border_color'],
        fill=styles['bg_color'],
        width=1
    )
    
    # Simuler une ombre
    shadow_offset = 3
    shadow_color = (0, 0, 0, 26)  # rgba(0,0,0,0.1) en valeurs RVB
    
    # Dessiner le contenu texte
    text_x = content_x + 20  # Marge interne
    text_y = content_y + 20  # Marge interne
    text_width = content_width - 40  # Largeur moins marges internes
    
    # Séparer le contenu en lignes
    lines = content.split('\n')
    current_y = text_y
    
    for line in lines:
        line = line.strip()
        if not line:
            current_y += text_font_size // 2
            continue
        
        # Vérifier s'il s'agit d'un titre h2
        if line.startswith('## '):
            line = line[3:]  # Supprimer le marqueur de titre
            draw.text((text_x, current_y), line, font=title_font, fill=styles['subtitle_color'])
            current_y += title_font_size + 10
        
        # Vérifier s'il s'agit d'un élément de liste
        elif line.startswith('- ') or line.startswith('* '):
            line = '• ' + line[2:]  # Remplacer le marqueur par un point
            draw.text((text_x + 10, current_y), line, font=text_font, fill=styles['text_color'])
            current_y += text_font_size + 5
        
        # Ignorer les lignes qui contiennent des URL d'images
        elif re.search(r'!\[.*?\]\(.*?\)', line):
            continue
        
        # Texte normal
        else:
            # Découper le texte pour qu'il tienne dans la largeur
            wrapped_lines = []
            words = line.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if text_font.getlength(test_line) <= text_width:
                    current_line = test_line
                else:
                    wrapped_lines.append(current_line)
                    current_line = word
            
            if current_line:
                wrapped_lines.append(current_line)
            
            # Dessiner chaque ligne
            for wrapped_line in wrapped_lines:
                draw.text((text_x, current_y), wrapped_line, font=text_font, fill=styles['text_color'])
                current_y += text_font_size + 5
        
        current_y += 5  # Espace supplémentaire entre les paragraphes
    
    # Ajouter l'image si présente
    if has_image and image_path and os.path.exists(image_path):
        try:
            # Charger l'image
            slide_image = Image.open(image_path)
            
            # Calculer les dimensions pour préserver le ratio
            img_width, img_height = slide_image.size
            img_ratio = img_height / img_width
            
            new_img_width = image_width - 40  # Marge interne
            new_img_height = int(new_img_width * img_ratio)
            
            # Limiter la hauteur
            max_height = content_height - 40  # Marge interne
            if new_img_height > max_height:
                new_img_height = max_height
                new_img_width = int(new_img_height / img_ratio)
            
            # Redimensionner l'image
            slide_image = slide_image.resize((new_img_width, new_img_height), Image.LANCZOS)
            
            # Créer un cadre pour l'image
            image_container_x = image_x
            image_container_y = content_y
            image_container_width = image_width
            image_container_height = content_height
            
            # Dessiner le cadre de l'image
            draw.rounded_rectangle(
                [(image_container_x, image_container_y), 
                 (image_container_x + image_container_width, image_container_y + image_container_height)],
                radius=border_radius,
                outline=styles['border_color'],
                fill=styles['image_container_bg'],
                width=1
            )
            
            # Positionner l'image au centre du conteneur
            image_position_x = image_container_x + (image_container_width - new_img_width) // 2
            image_position_y = image_container_y + (image_container_height - new_img_height) // 2
            
            # Coller l'image
            img.paste(slide_image, (image_position_x, image_position_y))
        except Exception as e:
            print(f" Erreur lors du traitement de l'image {image_path}: {e}")
    
    # Générer un nom de fichier si non fourni
    if not output_path:
        output_dir = "C:/Users/ThinkPad/Desktop/plateform/python/html_styled_slides"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"slide_{slide_num:03d}.png")
    
    # Sauvegarder l'image
    img.save(output_path)
    
    return output_path
def extract_content_from_pdf(pdf_path):
    """
    Extrait le texte et les images d'un PDF, format compatible avec le reste du code
    
    Args:
        pdf_path: Chemin du fichier PDF
        
    Returns:
        Une liste de tuples (titre, texte de la page, chemin de l'image si présente)
    """
    print(f" Extraction du contenu de: {pdf_path}")
    
    try:
        # Importer les bibliothèques nécessaires
        import PyPDF2
        import fitz  # PyMuPDF
        from pathlib import Path
        
        print(f" Extraction du texte et des images du PDF...")
        
        # Répertoire pour les images extraites
        image_dir = Path("./extracted_images").absolute()
        image_dir.mkdir(exist_ok=True)
        
        # Extraire le texte de chaque page
        page_texts = []
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            for page_num in range(num_pages):
                # Extraire le texte de la page
                page = reader.pages[page_num]
                page_text = page.extract_text() or ""
                page_texts.append(page_text)
        
        # Ouvrir le PDF pour extraire les images
        doc = fitz.open(pdf_path)
        content_items = []
        
        # Pour chaque page, extraire le texte et l'image
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page_texts[page_num]
            
            # Générer un titre simple pour la page
            title = f"Page {page_num + 1}"
            
            # Chercher des images significatives
            image_path = None
            image_list = page.get_images(full=True)
            
            # Si la page contient des images significatives
            if image_list:
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        
                        # Vérifier si l'image est significative (pas un logo minuscule)
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)
                        
                        if width < 100 or height < 100:
                            continue
                            
                        # Extraire l'image
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_filename = f"image_{page_num+1}_{img_index+1}.{image_ext}"
                        image_path = image_dir / image_filename
                        
                        # Sauvegarder l'image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # On a trouvé une image significative, on arrête la recherche
                        image_path = str(image_path)
                        break
                        
                    except Exception as e:
                        print(f" Erreur extraction image: {e}")
            
            # Si aucune image significative, capturer la page entière comme image optionnelle
            if not image_path:
                try:
                    # Capturer la page comme image uniquement pour la première page ou pages importantes
                    if page_num == 0 or (text and len(text.strip()) > 50):
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        image_filename = f"page_{page_num+1}.png"
                        image_path = image_dir / image_filename
                        pix.save(str(image_path))
                        image_path = str(image_path)
                except Exception as e:
                    print(f" Erreur capture page: {e}")
            
            # Ajouter le contenu de la page (titre, texte et image) à la liste
            content_items.append((title, text, image_path))
        
        print(f" Extraction PDF terminée: {len(content_items)} pages traitées")
        return content_items
        
    except ImportError as e:
        print(f"PyPDF2 et PyMuPDF sont nécessaires. Erreur: {e}")
        print("Installez-les avec: pip install PyPDF2 PyMuPDF")
        return []
    except Exception as e:
        print(f" Erreur lors du traitement du PDF: {e}")
        import traceback
        traceback.print_exc()
        return []
def extract_slides_from_markdown(markdown_path):
    """
    Extrait uniquement le texte et les images spécifiques des fichiers PDF ou Markdown,
    sans créer de diapositives complètes
    
    Args:
        markdown_path: Chemin du fichier PDF ou Markdown
        
    Returns:
        Une liste de tuples (titre, contenu, image)
    """
    print(f" Extraction du contenu de: {markdown_path}")
    
    # Vérifier si c'est un fichier PDF
    if markdown_path.lower().endswith('.pdf'):
        try:
            # Importer les bibliothèques nécessaires            
            print(f" Extraction du texte et des images spécifiques du PDF...")
            
            # Répertoire pour les images extraites
            image_dir = Path("./extracted_images").absolute()
            image_dir.mkdir(exist_ok=True)
            
            # Extraire le texte de chaque page
            page_texts = []
            with open(markdown_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                for page_num in range(num_pages):
                    # Extraire le texte de la page
                    page = reader.pages[page_num]
                    page_text = page.extract_text() or ""
                    page_texts.append(page_text)
            
            # Ouvrir le PDF pour extraire uniquement les images spécifiques
            doc = fitz.open(markdown_path)
            content_items = []
            
            # Pour chaque page, extraire le texte et les images spécifiques
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page_texts[page_num]
                
                # Générer un titre simple pour la page
                title = f"Page {page_num + 1}"
                
                # Chercher uniquement des images significatives intégrées
                image_path = None
                image_list = page.get_images(full=True)
                
                # Si la page contient des images intégrées
                if image_list:
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            
                            # Vérifier si l'image est significative (pas un logo minuscule)
                            width = base_image.get("width", 0)
                            height = base_image.get("height", 0)
                            
                            if width < 100 or height < 100:
                                continue
                                
                            # Extraire l'image
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            image_filename = f"image_{page_num+1}_{img_index+1}.{image_ext}"
                            image_path = image_dir / image_filename
                            
                            # Sauvegarder l'image
                            with open(image_path, "wb") as img_file:
                                img_file.write(image_bytes)
                            
                            # On a trouvé une image significative
                            image_path = str(image_path)
                            break
                            
                        except Exception as e:
                            print(f" Erreur extraction image: {e}")
            
                # Ajouter le contenu de la page (titre, texte et image) à la liste
                content_items.append((title, text, image_path))
        
            print(f" Extraction PDF terminée: {len(content_items)} pages traitées")
            return content_items
            
        except ImportError as e:
            print(f"PyPDF2 et PyMuPDF sont nécessaires. Erreur: {e}")
            print("Installez-les avec: pip install PyPDF2 PyMuPDF")
            return []
        except Exception as e:
            print(f" Erreur lors du traitement du PDF: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # Traitement des fichiers markdown
    else:
        try:
            # Lire le contenu du fichier
            try:
                with open(markdown_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                try:
                    with open(markdown_path, 'r', encoding='latin-1') as file:
                        content = file.read()
                    print("⚠️ Fichier encodé en latin-1, conversion effectuée")
                except Exception as e:
                    print(f" Erreur de lecture du fichier: {e}")
                    return []
            
            # Extraire uniquement le texte et les images spécifiques, sans créer de diapositives
            # Diviser le contenu en sections (si présentes)
            content_items = []
            
            # Essayer de diviser par les marqueurs de page ou par les titres
            page_markers = re.findall(r'##\s+---\s+Page\s+\d+\s+---', content)
            
            if page_markers:
                # Diviser par marqueurs de page
                sections = re.split(r'##\s+---\s+Page\s+\d+\s+---', content)
                
                for i, section in enumerate(sections):
                    if not section.strip():
                        continue
                    
                    # Utiliser les numéros de page comme titres
                    if i < len(page_markers):
                        title = page_markers[i].strip('# -')
                    else:
                        title = f"Section {i+1}"
                    
                    # Extraire les images de cette section
                    image_path = None
                    image_matches = re.findall(r'!\[(.*?)\]\((.*?)\)', section)
                    
                    for alt_text, img_path in image_matches:
                        # Essayer différents chemins
                        found_path = None
                        possible_paths = [
                            img_path,
                            os.path.join(os.path.dirname(markdown_path), img_path),
                            f"C:/Users/ThinkPad/Desktop/plateform/python/{img_path}",
                        ]
                        
                        for path in possible_paths:
                            if os.path.exists(path):
                                found_path = path
                                break
                        
                        if found_path:
                            image_path = found_path
                            break
                    
                    # Supprimer les références d'images du texte
                    text = re.sub(r'!\[.*?\]\(.*?\)', '', section).strip()
                    
                    # Ajouter le contenu extrait
                    content_items.append((title, text, image_path))
            else:
                # Pas de marqueurs de page, chercher les titres
                headers = re.findall(r'^(#+)\s+(.*?)$', content, re.MULTILINE)
                
                if headers:
                    # Diviser par titres
                    title_pattern = r'^#+\s+.*?$'
                    sections = re.split(title_pattern, content, flags=re.MULTILINE)
                    
                    for i, section in enumerate(sections):
                        if not section.strip():
                            continue
                        
                        # Utiliser les titres détectés
                        title = headers[i-1][1] if i > 0 and i-1 < len(headers) else f"Section {i+1}"
                        
                        # Extraire les images de cette section
                        image_path = None
                        image_matches = re.findall(r'!\[(.*?)\]\((.*?)\)', section)
                        
                        for alt_text, img_path in image_matches:
                            # Essayer différents chemins
                            found_path = None
                            possible_paths = [
                                img_path,
                                os.path.join(os.path.dirname(markdown_path), img_path),
                                f"C:/Users/ThinkPad/Desktop/plateform/python/{img_path}",
                            ]
                            
                            for path in possible_paths:
                                if os.path.exists(path):
                                    found_path = path
                                    break
                            
                            if found_path:
                                image_path = found_path
                                break
                        
                        # Supprimer les références d'images du texte
                        text = re.sub(r'!\[.*?\]\(.*?\)', '', section).strip()
                        
                        # Ajouter le contenu extrait
                        content_items.append((title, text, image_path))
                else:
                    # Pas de structure claire, traiter comme un document unique
                    title = "Document"
                    
                    # Extraire les images
                    image_path = None
                    image_matches = re.findall(r'!\[(.*?)\]\((.*?)\)', content)
                    
                    for alt_text, img_path in image_matches:
                        # Essayer différents chemins
                        found_path = None
                        possible_paths = [
                            img_path,
                            os.path.join(os.path.dirname(markdown_path), img_path),
                            f"C:/Users/ThinkPad/Desktop/plateform/python/{img_path}",
                        ]
                        
                        for path in possible_paths:
                            if os.path.exists(path):
                                found_path = path
                                break
                        
                        if found_path:
                            image_path = found_path
                            break
                    
                    # Supprimer les références d'images du texte
                    text = re.sub(r'!\[.*?\]\(.*?\)', '', content).strip()
                    
                    # Ajouter le contenu extrait
                    content_items.append((title, text, image_path))
            
            print(f" Extraction Markdown terminée: {len(content_items)} sections traitées")
            return content_items
            
        except Exception as e:
            print(f" Erreur lors du traitement du Markdown: {e}")
            import traceback
            traceback.print_exc()
            return []
def create_video_from_slides(slide_paths, output_path, duration=3, fps=24, transition=0.5):
    """
    Crée une vidéo à partir des diapositives
    
    Cette fonction utilise moviepy pour créer une vidéo à partir des images
    """
    try:
        from moviepy import ImageClip, concatenate_videoclips
        
        # Créer un clip pour chaque diapositive
        clips = []
        
        for slide_path in slide_paths:
            clip = ImageClip(slide_path).set_duration(duration)
            clips.append(clip)
        
        # Concaténer les clips
        video = concatenate_videoclips(clips, method="compose")
        
        # Sauvegarder la vidéo
        video.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            audio_codec='aac'
        )
        
        # Fermer les clips
        for clip in clips:
            clip.close()
        video.close()
        
        return True
    except Exception as e:
        print(f" Erreur lors de la création de la vidéo: {e}")
        return False

def main():
    # Créer le parseur d'arguments
    parser = argparse.ArgumentParser(description="Crée des diapositives et vidéos dans le style HTML")
    parser.add_argument("markdown_file", nargs="?", default="C:/Users/ThinkPad/Desktop/plateform/python/eca1.md", 
                        help="Chemin du fichier markdown (par défaut: C:/Users/ThinkPad/Desktop/plateform/python/eca1.md)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Durée d'affichage de chaque diapositive (par défaut: 3.0 secondes)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Images par seconde (par défaut: 24)")
    parser.add_argument("--slides-only", action="store_true",
                        help="Générer uniquement les diapositives, pas la vidéo")
    
    # Analyser les arguments
    args = parser.parse_args()
    
    # Vérifier si le fichier existe
    markdown_path = args.markdown_file
    if not os.path.exists(markdown_path):
        print(f" Fichier non trouvé: {markdown_path}")
        return
    
    # Créer les dossiers de sortie
    base_name = os.path.basename(markdown_path).split(".")[0]
    output_dir = create_directory(f"C:/Users/ThinkPad/Desktop/plateform/python/html_styled_slides/{base_name}")
    video_dir = create_directory("C:/Users/ThinkPad/Desktop/plateform/python/html_styled_videos")
    
    # Extraire les diapositives du fichier markdown
    slides = extract_slides_from_markdown(markdown_path)
    
    if not slides:
        print(" Aucune diapositive extraite")
        return
    
    # Créer les diapositives dans le style HTML
    slide_paths = []
    
    print(f" Création des diapositives stylisées...")
    for i, (title, content, image_path) in enumerate(slides):
        print(f"\r Création: {i+1}/{len(slides)} diapositives...", end="")
        
        # Créer la diapositive
        slide_path = create_html_style_slide(
            title,
            content,
            image_path,
            os.path.join(output_dir, f"slide_{i:03d}.png"),
            i
        )
        
        # Ajouter le chemin à la liste
        slide_paths.append(slide_path)
    
    print(f"\n {len(slide_paths)} diapositives créées dans: {output_dir}")
    
    # Créer la vidéo si demandé
    if not args.slides_only:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(markdown_path).split(".")[0]
        video_path = os.path.join(video_dir, f"{base_name}_html_styled_{timestamp}.mp4")
        
        print(f" Création de la vidéo: {video_path}")
        success = create_video_from_slides(slide_paths, video_path, args.duration, args.fps)
        
        if success:
            print(f" Vidéo créée avec succès: {video_path}")
        else:
            print(f" Échec de la création de la vidéo")
            print(f" Vous pouvez utiliser les diapositives dans: {output_dir}")
    else:
        print(f"ℹ Génération de vidéo désactivée")
        print(f" Vous pouvez utiliser les diapositives dans: {output_dir}")

if __name__ == "__main__":
    main()