import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, pipeline, ViTFeatureExtractor, VisionEncoderDecoderModel, AutoTokenizer, AutoModel
from transformers import AutoModelForSeq2SeqLM, AutoImageProcessor, AutoModelForImageClassification
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageEnhance, ImageOps, ImageStat
import re
import numpy as np

class TextProcessor:
    """
    Classe pour utiliser des modèles de langage avancés pour générer et améliorer des textes
    et créer des descriptions d'images
    """
    
    def __init__(self, model_path=None):
        self.model_path = model_path or "google/gemma-7b-it"  # Modèle par défaut
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.image_tokenizer = None
        self.image_model = None
        self.initialized = False
        
        # Configurer le prompt template selon le modèle
        self.prompt_template = "{prompt}"
        
        # Tentative d'initialisation du modèle
        self.initialize_model()
    
    def initialize_model(self):
        """Initialise le modèle et les pipelines nécessaires"""
        try:
            print(f"?? Initialisation du modèle de texte sur {self.device}...")
            
            # Déterminer quel modèle utiliser et configurer le prompt template approprié
            if "gemma" in self.model_path.lower():
                self.prompt_template = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            elif "llama" in self.model_path.lower():
                self.prompt_template = "<|system|>\nVous êtes un assistant expert en création de contenu.\n<|user|>\n{prompt}\n<|assistant|>\n"
            elif "mistral" in self.model_path.lower():
                self.prompt_template = "<s>[INST] {prompt} [/INST]"
            elif "phi" in self.model_path.lower():
                self.prompt_template = "Instruction: {prompt}\n\nResponse:"
            
            # Initialiser le tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except Exception as e:
                print(f"?? Erreur lors de l'initialisation du tokenizer standard, tentative avec LlamaTokenizer: {e}")
                self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
            
            # Initialiser le modèle avec quantification pour économiser la mémoire
            load_in_8bit = self.device == "cuda" and torch.cuda.get_device_properties(0).total_memory > 8e9
            load_in_4bit = self.device == "cuda" and not load_in_8bit
            
            # Choisir la configuration selon les ressources disponibles
            if load_in_8bit:
                print("?? Chargement du modèle en quantification 8-bit...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_8bit=True
                )
            elif load_in_4bit:
                print("?? Chargement du modèle en quantification 4-bit...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_4bit=True
                )
            else:
                print("?? Chargement du modèle en précision standard...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
            
            # Essayer de charger un modèle pour la description d'images
            try:
                print("??? Initialisation du modèle de description d'images...")
                
                # Utiliser un modèle plus léger de captioning
                image_model_name = "nlpconnect/vit-gpt2-image-captioning"
                
                self.image_processor = ViTFeatureExtractor.from_pretrained(image_model_name)
                self.image_tokenizer = AutoTokenizer.from_pretrained(image_model_name)
                self.image_model = VisionEncoderDecoderModel.from_pretrained(image_model_name)
                
                # Déplacer le modèle d'image sur le bon device
                if self.device == "cuda":
                    self.image_model = self.image_model.to("cuda")
                
                print("? Modèle de description d'images initialisé")
            except Exception as e:
                print(f"?? Impossible de charger le modèle de description d'images: {e}")
                self.image_processor = None
                self.image_model = None
            
            self.initialized = True
            print("? Modèle de texte initialisé avec succès!")
            
        except Exception as e:
            print(f"? Erreur lors de l'initialisation du modèle: {e}")
            traceback.print_exc()
            
            # Fallback vers un modèle plus petit et plus simple si disponible
            try:
                print("?? Tentative de fallback vers un modèle plus léger (GPT-2)...")
                self.model_path = "gpt2"
                self.prompt_template = "{prompt}"
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
                
                # Créer un pipeline simple
                self.text_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
                
                self.initialized = True
                print("? Modèle de fallback initialisé avec succès!")
            except:
                self.initialized = False
                print("? Échec de l'initialisation du modèle de fallback")
    
    def generate_text(self, prompt, max_length=512, temperature=0.7):
        """Génère du texte à partir d'une invite"""
        if not self.initialized:
            print("?? Modèle non initialisé, utilisation de texte par défaut")
            return "Le modèle n'a pas pu être chargé pour générer du texte."
        
        try:
            # Formater le prompt selon le template du modèle
            formatted_prompt = self.prompt_template.format(prompt=prompt)
            
            # Générer le texte soit avec le pipeline soit directement avec le modèle
            if hasattr(self, 'text_pipeline'):
                # Utiliser le pipeline si disponible (fallback GPT-2)
                generated_text = self.text_pipeline(
                    formatted_prompt,
                    max_length=len(formatted_prompt.split()) + max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    num_return_sequences=1
                )[0]['generated_text']
                
                # Extraire uniquement la réponse (sans le prompt)
                response = generated_text[len(formatted_prompt):].strip()
            else:
                # Utiliser directement le modèle
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Générer la réponse
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.95,
                        num_return_sequences=1
                    )
                
                # Décoder la réponse
                decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extraire la partie générée (après le prompt)
                # La méthode exacte dépend du modèle et du format de réponse
                if "gemma" in self.model_path.lower():
                    response = decoded_output.split("<start_of_turn>model")[-1].strip()
                elif "llama" in self.model_path.lower():
                    response = decoded_output.split("<|assistant|>")[-1].strip()
                elif "mistral" in self.model_path.lower():
                    response = decoded_output.split("[/INST]")[-1].strip()
                elif "phi" in self.model_path.lower():
                    response = decoded_output.split("Response:")[-1].strip()
                else:
                    # Fallback générique - essayer de trouver la partie après le prompt
                    response = decoded_output[len(formatted_prompt):].strip()
            
            return response
        except Exception as e:
            print(f"?? Erreur lors de la génération de texte: {e}")
            traceback.print_exc()
            return "Désolé, une erreur s'est produite lors de la génération du texte."
    
    def enhance_content(self, content, style="formel"):
        """Améliore le contenu textuel avec le modèle"""
        if not self.initialized:
            return content
        
        try:
            # Limiter la taille du contenu pour éviter les jetons trop longs
            if len(content) > 1000:
                truncated_content = content[:1000] + "..."
            else:
                truncated_content = content
            
            # Construire un prompt pour améliorer le texte
            prompt = f"""Améliore ce texte pour qu'il soit plus clair et plus {style}:
            
            "{truncated_content}"
            
            Version améliorée:"""
            
            # Générer le texte amélioré
            enhanced_text = self.generate_text(prompt, max_length=len(content) + 200)
            
            # Si le texte généré est trop court, garder l'original
            if len(enhanced_text) < len(content) / 2:
                print("?? Texte généré trop court, conservation du texte original")
                return content
            
            return enhanced_text
        except Exception as e:
            print(f"?? Erreur lors de l'amélioration du contenu: {e}")
            return content

    