"""
Chinese text conversion utilities.
Handles Simplified Chinese to Traditional Chinese conversion.
"""

from typing import List, Dict
import re

try:
    from opencc import OpenCC
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False
    print("Warning: OpenCC not available. Chinese conversion will be disabled.")


class ChineseConverter:
    """Converter for Simplified to Traditional Chinese."""
    
    def __init__(self):
        """Initialize the converter."""
        self.converter = None
        if OPENCC_AVAILABLE:
            try:
                # s2t: Simplified to Traditional
                # s2tw: Simplified to Traditional (Taiwan Standard)
                # s2hk: Simplified to Traditional (Hong Kong Standard)
                self.converter = OpenCC('s2tw')  # Using Taiwan standard
                print("✅ Chinese converter initialized (Simplified → Traditional TW)")
            except Exception as e:
                print(f"⚠️  Failed to initialize Chinese converter: {e}")
                self.converter = None
    
    def is_available(self) -> bool:
        """Check if converter is available."""
        return self.converter is not None
    
    def convert_text(self, text: str) -> str:
        """
        Convert Simplified Chinese to Traditional Chinese.
        
        Args:
            text: Input text (may contain Simplified Chinese)
        
        Returns:
            Converted text (Traditional Chinese)
        """
        if not self.converter:
            return text
        
        try:
            return self.converter.convert(text)
        except Exception as e:
            print(f"⚠️  Failed to convert text: {e}")
            return text
    
    def convert_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Convert all text in segments from Simplified to Traditional Chinese.
        
        Args:
            segments: List of segment dictionaries with 'text' field
        
        Returns:
            List of segments with converted text
        """
        if not self.converter:
            return segments
        
        converted_segments = []
        for seg in segments:
            converted_seg = seg.copy()
            if 'text' in converted_seg:
                converted_seg['text'] = self.convert_text(converted_seg['text'])
            converted_segments.append(converted_seg)
        
        return converted_segments


# Global converter instance
_converter = None


def get_converter() -> ChineseConverter:
    """Get or create the global converter instance."""
    global _converter
    if _converter is None:
        _converter = ChineseConverter()
    return _converter


def convert_to_traditional(text: str) -> str:
    """
    Convert Simplified Chinese to Traditional Chinese.
    
    Args:
        text: Input text
    
    Returns:
        Converted text
    """
    converter = get_converter()
    return converter.convert_text(text)


def convert_segments_to_traditional(segments: List[Dict]) -> List[Dict]:
    """
    Convert all text in segments to Traditional Chinese.
    
    Args:
        segments: List of segment dictionaries
    
    Returns:
        List of segments with Traditional Chinese text
    """
    converter = get_converter()
    return converter.convert_segments(segments)


def is_chinese_text(text: str) -> bool:
    """
    Check if text contains Chinese characters.
    
    Args:
        text: Input text
    
    Returns:
        True if text contains Chinese characters
    """
    # Unicode range for CJK Unified Ideographs
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(chinese_pattern.search(text))


if __name__ == "__main__":
    # Test the converter
    converter = ChineseConverter()
    
    if converter.is_available():
        # Test cases
        test_texts = [
            "这是简体中文",
            "欢迎使用语音识别系统",
            "Hello, 这是混合文本 with English",
            "繁體中文保持不變",
        ]
        
        print("\n📝 Testing Chinese Converter:")
        print("=" * 50)
        for text in test_texts:
            converted = converter.convert_text(text)
            print(f"原文: {text}")
            print(f"轉換: {converted}")
            print("-" * 50)
    else:
        print("❌ Converter not available")
