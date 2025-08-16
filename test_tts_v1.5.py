#!/usr/bin/env python3
"""
Test script for OLD IndexTTS infer_fast method (v1.5 behavior - single concatenated file).
Usage: python test_tts_v1.5.py
"""

from indextts.infer import IndexTTS

def main():
    # Initialize TTS model
    print(">> Initializing IndexTTS...")
    tts = IndexTTS(
        cfg_path="checkpoints/config.yaml", 
        model_dir="checkpoints", 
        is_fp16=True, 
        use_cuda_kernel=False
    )
    
    # Test text with multiple sentences - EXACT SAME as test_tts.py
    test_text = "会自己出账单，出帐单后会给你发邮件。你付了就行了，出帐单后有宽限期，大概两周左右。也就是收到邮件的两周内付款就行了。末节决战，韩国队发起疯狂反扑，一度将分差迫近至6分。关键时刻，韩国队主力内线五犯离场成为比赛转折点，随后胡金秋在内线大杀四方，连续得分稳住局势。最终，中国男篮顶住压力，以79:71力克老对手韩国队。时隔10年之后再次杀入亚洲杯四强。"
    
    # Run inference - will generate ONE concatenated file with OLD infer.py
    print(">> Running inference...")
    tts.infer_fast(
        audio_prompt="./reference.wav",  # Make sure this file exists
        text=test_text, 
        output_path="./output_v1.5.mp3",  # Single concatenated file output
        verbose=True
    )
    
    print(">> Done! Check output_v1.5.mp3 (single concatenated file)")

if __name__ == "__main__":
    main()