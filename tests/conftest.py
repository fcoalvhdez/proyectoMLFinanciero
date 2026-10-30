import sys
from pathlib import Path

# Permite importar scripts/ y flows/ desde las pruebas (el proyecto no se instala como paquete).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
