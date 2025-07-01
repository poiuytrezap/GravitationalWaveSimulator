from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn

from physics.gw_calculator import create_binary_system, GravitationalWaveCalculator

app = FastAPI(
    title="Gravitational Waves Simulator API",
    description="API pour simuler les ondes gravitationnelles de systèmes binaires",
    version="1.0.0"
)

# Configuration CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BinarySystemRequest(BaseModel):
    """Modèle pour la requête de système binaire"""
    m1: float = Field(..., ge=1.0, le=100.0, description="Masse 1 en masses solaires")
    m2: float = Field(..., ge=1.0, le=100.0, description="Masse 2 en masses solaires")
    separation: float = Field(..., ge=1e6, le=1e9, description="Séparation initiale en mètres")
    distance: float = Field(default=410.0, ge=1.0, le=10000.0, description="Distance en Mpc")
    inclination: float = Field(default=0.0, ge=0.0, le=180.0, description="Inclinaison en degrés")

class PresetRequest(BaseModel):
    """Modèle pour les systèmes prédéfinis"""
    preset_name: str = Field(..., description="Nom du preset (GW150914, GW170817, etc.)")
    distance: Optional[float] = Field(default=None, description="Distance personnalisée en Mpc")

# Systèmes prédéfinis basés sur les détections réelles
PRESETS = {
    "GW150914": {
        "m1": 36.0,
        "m2": 29.0,
        "separation": 3.5e8,  # Estimation à t=-0.2s
        "distance": 410.0,
        "description": "Première détection d'ondes gravitationnelles - Fusion de trous noirs"
    },
    "GW170817": {
        "m1": 1.17,
        "m2": 1.60,
        "separation": 1.2e8,
        "distance": 40.0,
        "description": "Kilonova - Fusion d'étoiles à neutrons avec contrepartie optique"
    },
    "GW190521": {
        "m1": 85.0,
        "m2": 66.0,
        "separation": 5e8,
        "distance": 5300.0,
        "description": "Fusion de trous noirs dans le 'mass gap'"
    },
    "EXTREME": {
        "m1": 50.0,
        "m2": 50.0,
        "separation": 1e8,
        "distance": 100.0,
        "description": "Système extrême pour démonstration"
    }
}

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Gravitational Waves Simulator API",
        "version": "1.0.0",
        "endpoints": [
            "/simulate",
            "/presets",
            "/preset/{preset_name}",
            "/calculate-properties"
        ]
    }

@app.get("/presets")
async def get_presets():
    """Retourne tous les systèmes prédéfinis disponibles"""
    return {
        "presets": PRESETS,
        "count": len(PRESETS)
    }

@app.get("/preset/{preset_name}")
async def get_preset(preset_name: str):
    """Retourne un système prédéfini spécifique"""
    if preset_name.upper() not in PRESETS:
        raise HTTPException(
            status_code=404, 
            detail=f"Preset '{preset_name}' non trouvé. Presets disponibles: {list(PRESETS.keys())}"
        )
    
    return {
        "preset": preset_name.upper(),
        "parameters": PRESETS[preset_name.upper()]
    }

@app.post("/simulate")
async def simulate_binary_system(request: BinarySystemRequest):
    """
    Simule l'évolution d'un système binaire et génère les ondes gravitationnelles
    """
    try:
        # Validation des paramètres
        if request.m1 + request.m2 > 200:
            raise HTTPException(
                status_code=400,
                detail="Masse totale trop élevée (max 200 masses solaires)"
            )
        
        # Calcul de la simulation
        result = create_binary_system(
            m1=request.m1,
            m2=request.m2,
            separation=request.separation,
            distance=request.distance
        )
        
        # Ajout des paramètres d'entrée pour référence
        result["input_parameters"] = {
            "m1": request.m1,
            "m2": request.m2,
            "separation": request.separation,
            "distance": request.distance,
            "inclination": request.inclination
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la simulation: {str(e)}"
        )

@app.post("/simulate-preset")
async def simulate_preset(request: PresetRequest):
    """
    Simule un système prédéfini
    """
    preset_name = request.preset_name.upper()
    
    if preset_name not in PRESETS:
        raise HTTPException(
            status_code=404,
            detail=f"Preset '{preset_name}' non trouvé"
        )
    
    preset_params = PRESETS[preset_name].copy()
    
    # Override distance si spécifiée
    if request.distance is not None:
        preset_params["distance"] = request.distance
    
    try:
        result = create_binary_system(**preset_params)
        result["preset_info"] = {
            "name": preset_name,
            "description": preset_params["description"]
        }
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la simulation du preset: {str(e)}"
        )

@app.post("/calculate-properties")
async def calculate_system_properties(m1: float, m2: float):
    """
    Calcule uniquement les propriétés du système sans simulation complète
    """
    try:
        calculator = GravitationalWaveCalculator()
        properties = calculator.calculate_merger_properties(m1, m2)
        
        return {
            "masses": {"m1": m1, "m2": m2},
            "properties": properties
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du calcul: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "gw-simulator"}

# Point d'entrée pour le développement
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
