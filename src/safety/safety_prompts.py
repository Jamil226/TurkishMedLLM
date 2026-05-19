#!/usr/bin/env python3
"""
Safety Prompts System for Turkish Medical LLM
Adds medical disclaimers, safety warnings, and ethical considerations to RAG responses
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class SafetyLevel(Enum):
    """Safety level configurations"""
    LOW = "low"           # Minimal disclaimers
    MEDIUM = "medium"     # Standard disclaimers
    HIGH = "high"         # Comprehensive safety warnings


@dataclass
class SafetyConfig:
    """Configuration for safety prompts"""
    level: SafetyLevel = SafetyLevel.HIGH
    include_emergency_warning: bool = True
    include_allergies_warning: bool = True
    include_professional_advice_disclaimer: bool = True
    include_dosage_warning: bool = True
    include_contraindication_warning: bool = True
    language: str = "turkish"  # turkish or english


class SafetyPrompts:
    """Turkish Medical Safety Prompts System"""
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.emergency_keywords = [
            'kalp krizi', 'stroke', 'şiddetli kanama', 'anafilaksi',
            'acil', 'hayati', 'tehlikeli', 'ciddi', 'ölümcül',
            'heart attack', 'stroke', 'severe bleeding', 'anaphylaxis'
        ]
        self.allergy_keywords = [
            'alerji', 'hipersensitivite', 'intolerans', 'reaksiyon',
            'allergy', 'hypersensitivity', 'intolerance', 'reaction'
        ]
    
    # Turkish Disclaimers
    TURKISH_PRE_DISCLAIMER_HIGH = """
⚠️ **TIBBİ UYARI - LÜTFEN OKUYUNUZ** ⚠️

Bu bilgiler genel eğitim amaçlıdır ve profesyonel tıbbi tavsiye değildir.

⚕️ **Önemli Sınırlamalar:**
• Bu yanıtlar bir doktor veya sağlık profesyonelinin yerini almaz
• Kişisel tıbbi tanı veya tedavi önerileri içermez
• Tüm medikal durumlar bireysaldir - mutlaka uzman danışınız
• Acil durumlarda derhal 112'yi arayın veya hastaneye gidin

---
"""

    TURKISH_PRE_DISCLAIMER_MEDIUM = """
⚠️ **Tıbbi Uyarı:** Bu bilgiler eğitim amaçlıdır. Profesyonel tıbbi tavsiye için sağlık profesyoneline başvurunuz.
"""

    TURKISH_PRE_DISCLAIMER_LOW = """
⚠️ Bilgilendirme amaçlıdır.
"""

    TURKISH_POST_DISCLAIMER_HIGH = """
---
⚕️ **SONUÇ UYARISI:**
• Bu bilgiler yaşam kurtarıcı olmayabilir
• Herhangi bir tıbbi karar vermeden önce bir doktora danışınız
• Yan etkiler hakkında eczacınıza sorunuz
• Belirsiz herhangi bir nokta için sağlık profesyoneline ulaşınız

**Acil Tıbbi Yardım:** 112 (Türkiye)
**Zehir İnformasyon Merkezi:** 114 (Türkiye)
"""

    TURKISH_POST_DISCLAIMER_MEDIUM = """
---
💡 **Önemli:** Sağlık profesyoneline danışmadan herhangi bir tedavi başlamayınız.
"""

    TURKISH_POST_DISCLAIMER_LOW = """
---
ℹ️ Tıbbi tavsiye için profesyonelle danışınız.
"""

    # English Disclaimers (for reference)
    ENGLISH_PRE_DISCLAIMER_HIGH = """
⚠️ **MEDICAL DISCLAIMER - PLEASE READ** ⚠️

This information is for educational purposes only and is NOT professional medical advice.

⚕️ **Important Limitations:**
• Does not replace consultation with a doctor or healthcare professional
• Does not provide personal medical diagnosis or treatment recommendations
• All medical conditions are individual - consult a specialist
• For emergencies, call emergency services (911 in US) or go to hospital immediately

---
"""

    ENGLISH_POST_DISCLAIMER_HIGH = """
---
⚕️ **IMPORTANT DISCLAIMER:**
• This information may not be life-saving
• Consult a doctor before making any medical decisions
• Ask your pharmacist about side effects
• Contact healthcare professional for any unclear information

**Emergency Medical Services:** 911 (USA)
**Poison Control:** 1-800-222-1222 (USA)
"""

    # Emergency Response
    TURKISH_EMERGENCY_RESPONSE = """
🚨 **ACIL DURUM - HEMEN YARDIM İSTEYİN!** 🚨

Bu durum potansiyel olarak yaşam tehlikesi oluşturabilir. 

**DERHAL YAPIN:**
1. ☎️ 112'yi arayın (ambulans talep edin)
2. 🏥 En yakın acil servise gidin
3. ⏰ Zaman kritiktir - beklemeyin!

Bu durum profesyonel tıbbi müdahale gerektirir.
"""

    ENGLISH_EMERGENCY_RESPONSE = """
🚨 **EMERGENCY - SEEK HELP IMMEDIATELY!** 🚨

This condition may be life-threatening.

**DO THIS NOW:**
1. ☎️ Call 911 or local emergency number
2. 🏥 Go to nearest emergency room
3. ⏰ Time is critical - don't wait!

This requires professional medical intervention.
"""

    # Allergy Warning
    TURKISH_ALLERGY_WARNING = """
⚠️ **ALERJİ UYARISI**

Herhangi bir ilaç veya tedavi başlamadan ÖNCE:
✓ Bilinen alerjilerinizi sağlık profesyoneline bildirin
✓ Aile alerjisi öyküsünü açıklayın
✓ Geçmiş reaksiyonlardan bahsedin

Alerjik reaksiyonun belirtileri:
• Kaşıntı, kızarıklık, şişme
• Nefes almada zorluk
• Hızlı kalp atışı
• Bulantı, kusma

Hızlı reaksiyonlar ölümcül olabilir → Derhal 112'yi arayın!
"""

    ENGLISH_ALLERGY_WARNING = """
⚠️ **ALLERGY WARNING**

Before starting any medication or treatment:
✓ Tell healthcare provider about all known allergies
✓ Mention family allergy history
✓ Describe past reactions

Signs of allergic reaction:
• Itching, redness, swelling
• Difficulty breathing
• Rapid heartbeat
• Nausea, vomiting

Severe reactions can be fatal → Call 911 immediately!
"""

    # Contraindication Warning
    TURKISH_CONTRAINDICATION_WARNING = """
⚠️ **KONTRENDIKASYON (YAPILMAMASI GEREKEN) UYARISI**

Bu durum veya tedavi aşağıdakilerle uyumsuz olabilir:
• Diğer ilaçlar (İLAÇ ETKİLEŞİMLERİ!)
• Kronik hastalıklar (kalp, böbrek, karaciğer vb.)
• Gebelik / Emzirme
• Yaş veya ağırlık kriterleri

**GEREKLİ İŞLEM:** Tüm ilaçlarınızı ve sağlık koşullarınızı doktora bildirin.
"""

    ENGLISH_CONTRAINDICATION_WARNING = """
⚠️ **CONTRAINDICATION (DO NOT USE) WARNING**

This condition or treatment may be incompatible with:
• Other medications (DRUG INTERACTIONS!)
• Chronic conditions (heart, kidney, liver disease, etc.)
• Pregnancy / Breastfeeding
• Age or weight criteria

**REQUIRED ACTION:** Tell doctor about all medications and health conditions.
"""

    # Dosage Warning
    TURKISH_DOSAGE_WARNING = """
⚠️ **DOZ UYARISI**

• Dozlar kişiye göre değişir (yaş, ağırlık, böbrek/karaciğer fonksiyonu)
• ASLA reçetesiz doz tavsiyesi yapılmaz
• İlaçlar kesin olarak doktor/eczacı tarafından belirtilmelidir
• Kendi doz ayarlaması yaparak hayatınızı tehlikeye atmayınız

Bu konuda kesinlikle profesyonel tavsiye alınız.
"""

    def add_emergency_warning(self, response: str) -> str:
        """Add emergency warning if keywords detected"""
        if not self.config.include_emergency_warning:
            return response
        
        response_lower = response.lower()
        for keyword in self.emergency_keywords:
            if keyword in response_lower:
                if self.config.language == "turkish":
                    return self.TURKISH_EMERGENCY_RESPONSE + "\n" + response
                else:
                    return self.ENGLISH_EMERGENCY_RESPONSE + "\n" + response
        
        return response

    def add_allergy_warning(self, response: str) -> str:
        """Add allergy warning if keywords detected"""
        if not self.config.include_allergies_warning:
            return response
        
        response_lower = response.lower()
        for keyword in self.allergy_keywords:
            if keyword in response_lower:
                if self.config.language == "turkish":
                    return response + "\n" + self.TURKISH_ALLERGY_WARNING
                else:
                    return response + "\n" + self.ENGLISH_ALLERGY_WARNING
        
        return response

    def add_pre_disclaimer(self, response: str) -> str:
        """Add pre-response disclaimer based on safety level"""
        if not self.config.include_professional_advice_disclaimer:
            return response
        
        if self.config.language != "turkish":
            if self.config.level == SafetyLevel.HIGH:
                disclaimer = self.ENGLISH_PRE_DISCLAIMER_HIGH
            elif self.config.level == SafetyLevel.MEDIUM:
                disclaimer = self.TURKISH_PRE_DISCLAIMER_MEDIUM
            else:
                disclaimer = self.ENGLISH_PRE_DISCLAIMER_HIGH
        else:
            if self.config.level == SafetyLevel.HIGH:
                disclaimer = self.TURKISH_PRE_DISCLAIMER_HIGH
            elif self.config.level == SafetyLevel.MEDIUM:
                disclaimer = self.TURKISH_PRE_DISCLAIMER_MEDIUM
            else:
                disclaimer = self.TURKISH_PRE_DISCLAIMER_LOW
        
        return disclaimer + response

    def add_post_disclaimer(self, response: str) -> str:
        """Add post-response disclaimer based on safety level"""
        if not self.config.include_professional_advice_disclaimer:
            return response
        
        if self.config.language != "turkish":
            if self.config.level == SafetyLevel.HIGH:
                disclaimer = self.ENGLISH_POST_DISCLAIMER_HIGH
            else:
                return response
        else:
            if self.config.level == SafetyLevel.HIGH:
                disclaimer = self.TURKISH_POST_DISCLAIMER_HIGH
            elif self.config.level == SafetyLevel.MEDIUM:
                disclaimer = self.TURKISH_POST_DISCLAIMER_MEDIUM
            else:
                disclaimer = self.TURKISH_POST_DISCLAIMER_LOW
        
        return response + disclaimer

    def add_dosage_warning(self, response: str) -> str:
        """Add dosage warning if dosage information detected"""
        if not self.config.include_dosage_warning:
            return response
        
        dosage_keywords = ['mg', 'ml', 'doz', 'dose', 'günde', 'perday', 'her', 'each']
        response_lower = response.lower()
        
        for keyword in dosage_keywords:
            if keyword in response_lower:
                if self.config.language == "turkish":
                    return response + "\n" + self.TURKISH_DOSAGE_WARNING
                else:
                    return response
        
        return response

    def add_contraindication_warning(self, response: str) -> str:
        """Add contraindication warning if needed"""
        if not self.config.include_contraindication_warning:
            return response
        
        contraindication_keywords = [
            'kontrendikasyon', 'contraindication', 'uyumsuz',
            'uyarı', 'dikkat', 'gibi durumlarda', 'when'
        ]
        response_lower = response.lower()
        
        for keyword in contraindication_keywords:
            if keyword in response_lower:
                if self.config.language == "turkish":
                    return response + "\n" + self.TURKISH_CONTRAINDICATION_WARNING
                else:
                    return response
        
        return response

    def process_response(self, response: str, add_all_warnings: bool = False) -> str:
        """
        Apply all appropriate safety prompts to response
        
        Args:
            response: Original RAG response
            add_all_warnings: If True, add all warnings regardless of keywords
        
        Returns:
            Response with safety prompts added
        """
        # Add pre-disclaimer
        result = self.add_pre_disclaimer(response)
        
        # Add emergency warning
        result = self.add_emergency_warning(result)
        
        # Add allergy warning
        result = self.add_allergy_warning(result)
        
        # Add dosage warning if present
        result = self.add_dosage_warning(result)
        
        # Add contraindication warning if present
        result = self.add_contraindication_warning(result)
        
        # Add post-disclaimer
        result = self.add_post_disclaimer(result)
        
        return result


def get_safety_config(level: str = "high") -> SafetyConfig:
    """
    Get standard safety configuration
    
    Args:
        level: "low", "medium", or "high"
    
    Returns:
        SafetyConfig object
    """
    level_map = {
        "low": SafetyLevel.LOW,
        "medium": SafetyLevel.MEDIUM,
        "high": SafetyLevel.HIGH
    }
    
    return SafetyConfig(level=level_map.get(level, SafetyLevel.HIGH))


if __name__ == "__main__":
    # Test examples
    print("=" * 80)
    print("🔬 Safety Prompts System - Test Examples")
    print("=" * 80)
    
    # Test response about diabetes
    test_response = """
Diyabet hastalarında kan şekeri yükselişinin ana nedenleri:

1. **İnsulin Eksikliği**: Pankreas yeterince insülin üretmediği için
2. **İnsulin Direnci**: Vücudun insülinin etkisine karşı direnç göstermesi
3. **Yanlış Beslenme**: Çok fazla şeker ve karbonhidrat tüketimi
4. **Hareketsiz Yaşam**: Fiziksel aktivite yetersizliği
5. **Stres**: Yüksek kortizol seviyeleri

Tedavi:
- 500 mg Metformin günde 2-3 kez
- Diyet kontrolü
- Düzenli egzersiz
"""
    
    # Test with HIGH safety level
    print("\n📋 TEST 1: HIGH Safety Level")
    print("-" * 80)
    safety_high = SafetyPrompts(SafetyConfig(level=SafetyLevel.HIGH))
    result_high = safety_high.process_response(test_response)
    print(result_high)
    
    # Test with MEDIUM safety level
    print("\n📋 TEST 2: MEDIUM Safety Level")
    print("-" * 80)
    safety_medium = SafetyPrompts(SafetyConfig(level=SafetyLevel.MEDIUM))
    result_medium = safety_medium.process_response(test_response)
    print(result_medium)
    
    # Test with emergency keywords
    print("\n📋 TEST 3: Emergency Detection")
    print("-" * 80)
    emergency_response = "Kalp krizi belirtileri: göğüs ağrısı, nefes almada zorluk..."
    safety_emergency = SafetyPrompts(SafetyConfig(level=SafetyLevel.HIGH))
    result_emergency = safety_emergency.process_response(emergency_response)
    print(result_emergency)
