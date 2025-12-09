import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
from datetime import datetime


class Generator(nn.Module):
    """
    Arquitectura del Generador para GAN.
    Ajusta las capas según tu modelo entrenado.
    """
    def __init__(self, latent_dim=100, img_channels=3, img_size=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Arquitectura básica - AJUSTA según tu modelo
        self.model = nn.Sequential(
            # Capa inicial: latent_dim -> 512 * 4 * 4
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 512 x 4 x 4 -> 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 256 x 8 x 8 -> 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 128 x 16 x 16 -> 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64 x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        # z: (batch_size, latent_dim, 1, 1)
        img = self.model(z)
        return img


# Variables globales para cachear el modelo
_generator = None
_device = None
_latent_dim = 100


def load_generator(model_path="./models/generator.pth", latent_dim=100, img_channels=3, img_size=64):
    """
    Carga el modelo generador desde un archivo .pth
    
    Args:
        model_path: ruta al archivo .pth del generador
        latent_dim: dimensión del vector latente
        img_channels: canales de la imagen (3 para RGB)
        img_size: tamaño de la imagen generada
    
    Returns:
        generator: modelo cargado
        device: dispositivo (cuda o cpu)
    """
    global _generator, _device, _latent_dim
    
    # Si ya está cargado, retornar
    if _generator is not None:
        return _generator, _device
    
    # Detectar dispositivo
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {_device}")
    
    # Crear modelo
    _generator = Generator(latent_dim=latent_dim, img_channels=img_channels, img_size=img_size)
    
    # Cargar pesos
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=_device)
    
    # Manejar diferentes formatos de guardado
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            _generator.load_state_dict(checkpoint['model_state_dict'])
        elif 'generator_state_dict' in checkpoint:
            _generator.load_state_dict(checkpoint['generator_state_dict'])
        elif 'state_dict' in checkpoint:
            _generator.load_state_dict(checkpoint['state_dict'])
        else:
            _generator.load_state_dict(checkpoint)
    else:
        _generator.load_state_dict(checkpoint)
    
    _generator.to(_device)
    _generator.eval()
    _latent_dim = latent_dim
    
    print(f"Modelo cargado exitosamente desde: {model_path}")
    return _generator, _device


def generate_portrait(
    output_dir="./images",
    model_path="./models/generator.pth",
    latent_dim=100,
    seed=None
):
    """
    Genera un retrato sintético usando el modelo GAN.
    
    Args:
        output_dir: directorio donde guardar la imagen
        model_path: ruta al modelo .pth
        latent_dim: dimensión del vector latente
        seed: semilla para reproducibilidad (opcional)
    
    Returns:
        str: ruta del archivo de imagen generado
    """
    # Cargar modelo (usa caché si ya está cargado)
    generator, device = load_generator(model_path, latent_dim)
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Establecer semilla si se proporciona
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Generar vector latente aleatorio
    with torch.no_grad():
        z = torch.randn(1, latent_dim, 1, 1, device=device)
        fake_img = generator(z)
    
    # Generar nombre único basado en timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"retrato_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Guardar imagen
    # save_image normaliza de [-1, 1] a [0, 1] automáticamente
    save_image(fake_img, filepath, normalize=True)
    
    print(f"Retrato generado: {filepath}")
    return filepath

if __name__ == "__main__":
    # Ejemplo de uso
    print("Generando retrato de prueba...")
    path = generate_portrait()
    print(f"Retrato guardado en: {path}")