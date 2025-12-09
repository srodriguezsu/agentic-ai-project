import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
from datetime import datetime


class Generator(nn.Module):
    """
    Arquitectura del Generador DCGAN - Coincide con el modelo de entrenamiento
    """
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Entrada: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Estado: 512 x 4 x 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Estado: 256 x 8 x 8
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Estado: 128 x 16 x 16
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Estado: 64 x 32 x 32
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Salida: 3 x 64 x 64
        )
    
    def forward(self, z):
        return self.main(z)


# Variables globales para cachear el modelo
_generator = None
_device = None
_latent_dim = 100


def load_generator(model_path="generator.pth", latent_dim=100):
    """
    Carga el modelo generador desde un archivo .pth
    
    Args:
        model_path: ruta al archivo .pth del generador (por defecto "generator.pth")
        latent_dim: dimensión del vector latente (debe ser 100)
    
    Returns:
        generator: modelo cargado
        device: dispositivo (cuda o cpu)
    """
    global _generator, _device, _latent_dim
    
    # Si ya está cargado, retornar
    if _generator is not None:
        print(f"Usando modelo ya cargado en memoria ({_device})")
        return _generator, _device
    
    # Detectar dispositivo
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {_device}")
    
    # Crear modelo con la arquitectura correcta
    _generator = Generator(latent_dim=latent_dim)
    
    # Cargar pesos
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo en: {model_path}\n"
            f"Asegúrate de que el archivo 'generator.pth' existe en el directorio actual."
        )
    
    try:
        # Cargar el state_dict directamente
        state_dict = torch.load(model_path, map_location=_device)
        _generator.load_state_dict(state_dict)
        _generator.to(_device)
        _generator.eval()
        _latent_dim = latent_dim
        
        print(f"✓ Modelo cargado exitosamente desde: {model_path}")
        return _generator, _device
    
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo: {str(e)}")


def generate_portrait(
    output_dir="generated_portraits",
    model_path="generator.pth",
    latent_dim=100,
    seed=None,
    num_images=1
):
    """
    Genera uno o más retratos sintéticos usando el modelo GAN.
    
    Args:
        output_dir: directorio donde guardar las imágenes
        model_path: ruta al modelo .pth (por defecto "generator.pth")
        latent_dim: dimensión del vector latente (debe ser 100)
        seed: semilla para reproducibilidad (opcional)
        num_images: número de imágenes a generar
    
    Returns:
        list: rutas de los archivos de imágenes generados
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
        print(f"Usando semilla: {seed}")
    
    generated_paths = []
    
    print(f"Generando {num_images} retrato(s)...")
    
    for i in range(num_images):
        # Generar vector latente aleatorio
        with torch.no_grad():
            z = torch.randn(1, latent_dim, 1, 1, device=device)
            fake_img = generator(z)
        
        # Generar nombre único basado en timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"retrato_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Guardar imagen
        # save_image normaliza de [-1, 1] a [0, 1] automáticamente
        save_image(fake_img, filepath, normalize=True)
        
        generated_paths.append(filepath)
        print(f"  [{i+1}/{num_images}] ✓ {filepath}")
    
    return generated_paths


def generate_batch(
    num_images=16,
    output_path="batch_portraits.png",
    model_path="generator.pth",
    latent_dim=100,
    seed=None,
    nrow=4
):
    """
    Genera un grid de múltiples retratos en una sola imagen.
    
    Args:
        num_images: número de retratos a generar
        output_path: ruta donde guardar el grid
        model_path: ruta al modelo .pth
        latent_dim: dimensión del vector latente
        seed: semilla para reproducibilidad
        nrow: número de imágenes por fila en el grid
    
    Returns:
        str: ruta del archivo generado
    """
    # Cargar modelo
    generator, device = load_generator(model_path, latent_dim)
    
    # Establecer semilla si se proporciona
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    print(f"Generando grid de {num_images} retratos...")
    
    # Generar batch de imágenes
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, 1, 1, device=device)
        fake_imgs = generator(z)
    
    # Guardar como grid
    save_image(fake_imgs, output_path, nrow=nrow, normalize=True, padding=2)
    
    print(f"✓ Grid guardado en: {output_path}")
    return output_path


# Función para explorar el espacio latente
def interpolate_portraits(
    steps=10,
    output_dir="interpolation",
    model_path="generator.pth",
    latent_dim=100,
    seed1=None,
    seed2=None
):
    """
    Genera una interpolación entre dos retratos aleatorios.
    
    Args:
        steps: número de pasos en la interpolación
        output_dir: directorio donde guardar las imágenes
        model_path: ruta al modelo .pth
        latent_dim: dimensión del vector latente
        seed1: semilla para el primer retrato
        seed2: semilla para el segundo retrato
    
    Returns:
        list: rutas de los archivos generados
    """
    # Cargar modelo
    generator, device = load_generator(model_path, latent_dim)
    
    # Crear directorio
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar vectores latentes inicial y final
    if seed1 is not None:
        torch.manual_seed(seed1)
    z1 = torch.randn(1, latent_dim, 1, 1, device=device)
    
    if seed2 is not None:
        torch.manual_seed(seed2)
    z2 = torch.randn(1, latent_dim, 1, 1, device=device)
    
    print(f"Generando interpolación con {steps} pasos...")
    
    generated_paths = []
    
    with torch.no_grad():
        for i, alpha in enumerate(torch.linspace(0, 1, steps)):
            # Interpolación lineal
            z = (1 - alpha) * z1 + alpha * z2
            fake_img = generator(z)
            
            # Guardar imagen
            filename = f"interpolation_step_{i:03d}.png"
            filepath = os.path.join(output_dir, filename)
            save_image(fake_img, filepath, normalize=True)
            
            generated_paths.append(filepath)
            print(f"  [{i+1}/{steps}] ✓ {filepath}")
    
    return generated_paths