import numpy as np
from numba import jit
from typing import Tuple, Dict, List
import math

class GravitationalWaveCalculator:
    """
    Calculateur d'ondes gravitationnelles pour systèmes binaires
    Utilise l'approximation post-newtonienne pour les orbites
    et la formule de quadrupole pour l'émission d'ondes
    """
    
    def __init__(self):
        # Constantes physiques (unités géométriques G=c=1)
        self.G = 6.67430e-11  # m³/kg/s²
        self.c = 299792458    # m/s
        self.Msun = 1.98847e30  # kg
        
        # Constantes dérivées
        self.G_over_c3 = self.G / (self.c**3)
        self.c_over_G = self.c / self.G
        
    def calculate_chirp_mass(self, m1: float, m2: float) -> float:
        """
        Calcule la masse chirp du système binaire
        m1, m2 en masses solaires
        """
        total_mass = m1 + m2
        reduced_mass = (m1 * m2) / total_mass
        return (reduced_mass**(3/5)) * (total_mass**(2/5))
    
    def calculate_symmetric_mass_ratio(self, m1: float, m2: float) -> float:
        """
        Calcule le rapport de masses symétriques η
        """
        total_mass = m1 + m2
        return (m1 * m2) / (total_mass**2)
    
    @jit(nopython=True)
    def orbital_evolution(self, m1: float, m2: float, initial_separation: float, 
                         duration: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule l'évolution orbitale due à l'émission d'ondes gravitationnelles
        
        Returns:
        - times: array des temps
        - separations: array des séparations
        - frequencies: array des fréquences orbitales
        """
        # Conversion en unités SI
        m1_kg = m1 * 1.98847e30
        m2_kg = m2 * 1.98847e30
        total_mass = m1_kg + m2_kg
        reduced_mass = (m1_kg * m2_kg) / total_mass
        
        # Arrays pour stocker les résultats
        n_steps = int(duration / dt)
        times = np.zeros(n_steps)
        separations = np.zeros(n_steps)
        frequencies = np.zeros(n_steps)
        
        # Conditions initiales
        r = initial_separation
        
        for i in range(n_steps):
            times[i] = i * dt
            separations[i] = r
            
            # Fréquence orbitale (3ème loi de Kepler)
            omega = math.sqrt(6.67430e-11 * total_mass / (r**3))
            frequencies[i] = omega / (2 * math.pi)
            
            # Taux de perte d'énergie (formule de quadrupole)
            # dE/dt = -32/5 * G⁴/c⁵ * (m1*m2)²*(m1+m2)/r⁵
            energy_loss_rate = (-32/5) * (6.67430e-11**4) / (299792458**5) * \
                              (reduced_mass**2) * total_mass / (r**5)
            
            # Énergie orbitale
            orbital_energy = -6.67430e-11 * m1_kg * m2_kg / (2 * r)
            
            # dr/dt depuis dE/dt
            dr_dt = (2 * energy_loss_rate * r**2) / (6.67430e-11 * m1_kg * m2_kg)
            
            # Mise à jour de la séparation
            r += dr_dt * dt
            
            # Arrêt si collision (horizon de Schwarzschild approximatif)
            if r < 2 * 6.67430e-11 * total_mass / (299792458**2):
                # Truncate arrays
                times = times[:i+1]
                separations = separations[:i+1]
                frequencies = frequencies[:i+1]
                break
        
        return times, separations, frequencies
    
    def generate_waveform(self, m1: float, m2: float, distance: float, 
                         times: np.ndarray, separations: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Génère la forme d'onde gravitationnelle h(t)
        
        Args:
        - m1, m2: masses en masses solaires
        - distance: distance à la source en Mpc
        - times, separations: évolution orbitale
        
        Returns:
        - Dictionary avec h_plus, h_cross, strain_amplitude
        """
        # Conversion en unités SI
        m1_kg = m1 * self.Msun
        m2_kg = m2 * self.Msun
        total_mass = m1_kg + m2_kg
        reduced_mass = (m1_kg * m2_kg) / total_mass
        distance_m = distance * 3.086e22  # Mpc to meters
        
        # Calcul de l'amplitude du strain
        strain_amplitude = np.zeros(len(times))
        phase = np.zeros(len(times))
        
        for i in range(len(times)):
            r = separations[i]
            
            # Fréquence orbitale
            omega = np.sqrt(self.G * total_mass / (r**3))
            
            # Amplitude du strain (approximation de quadrupole)
            h0 = (4 * self.G**2 / (self.c**4 * distance_m)) * \
                 (reduced_mass * total_mass) / (r**2)
            
            strain_amplitude[i] = h0
            
            # Phase (intégration de la fréquence)
            if i > 0:
                phase[i] = phase[i-1] + 2 * omega * (times[i] - times[i-1])
        
        # Polarisations
        h_plus = strain_amplitude * (1 + np.cos(0)**2) / 2 * np.cos(2 * phase)
        h_cross = strain_amplitude * np.cos(0) * np.sin(2 * phase)
        
        return {
            'times': times,
            'h_plus': h_plus,
            'h_cross': h_cross,
            'strain_amplitude': strain_amplitude,
            'phase': phase,
            'frequency': np.gradient(phase) / (2 * np.pi * np.gradient(times))
        }
    
    def calculate_merger_properties(self, m1: float, m2: float) -> Dict[str, float]:
        """
        Calcule les propriétés de la fusion
        """
        total_mass = m1 + m2
        chirp_mass = self.calculate_chirp_mass(m1, m2)
        eta = self.calculate_symmetric_mass_ratio(m1, m2)
        
        # Fréquence ISCO (Innermost Stable Circular Orbit)
        # Pour trous noirs non-rotatifs
        r_isco = 6 * self.G * total_mass * self.Msun / (self.c**2)
        f_isco = 1 / (2 * np.pi) * np.sqrt(self.G * total_mass * self.Msun / (r_isco**3))
        
        # Temps jusqu'à la fusion (approximation)
        t_merger = (5/256) * (self.c**5 / self.G**(5/3)) * \
                   (chirp_mass * self.Msun)**(-5/3) * (np.pi * f_isco)**(-8/3)
        
        return {
            'total_mass': total_mass,
            'chirp_mass': chirp_mass,
            'symmetric_mass_ratio': eta,
            'isco_frequency': f_isco,
            'merger_time': t_merger,
            'final_mass': total_mass * (1 - 0.05 * eta)  # Approximation simple
        }

# Fonctions utilitaires pour l'API
def create_binary_system(m1: float, m2: float, separation: float, 
                        distance: float = 410) -> Dict:
    """
    Crée un système binaire et calcule son évolution
    """
    calculator = GravitationalWaveCalculator()
    
    # Calcul des propriétés du système
    properties = calculator.calculate_merger_properties(m1, m2)
    
    # Évolution orbitale
    duration = min(properties['merger_time'], 10.0)  # Max 10 secondes
    dt = duration / 1000  # 1000 points
    
    times, separations, frequencies = calculator.orbital_evolution(
        m1, m2, separation, duration, dt
    )
    
    # Génération de la forme d'onde
    waveform = calculator.generate_waveform(m1, m2, distance, times, separations)
    
    return {
        'system_properties': properties,
        'evolution': {
            'times': times.tolist(),
            'separations': separations.tolist(),
            'frequencies': frequencies.tolist()
        },
        'waveform': {
            'times': waveform['times'].tolist(),
            'h_plus': waveform['h_plus'].tolist(),
            'h_cross': waveform['h_cross'].tolist(),
            'frequency': waveform['frequency'].tolist()
        }
    }
