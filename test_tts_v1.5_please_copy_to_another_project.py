#!/usr/bin/env python3
"""
Test script for OLD IndexTTS with command line options.
Usage: 
    python test_tts_v1.5.py                    # fast mode
    python test_tts_v1.5.py --slow             # slow mode  
    python test_tts_v1.5.py --slow --verbose   # slow mode with verbose
"""

import argparse
import time
from indextts.infer import IndexTTS

def main():
    parser = argparse.ArgumentParser(description="IndexTTS v1.5 Test Script")
    parser.add_argument("--slow", action="store_true", help="Use slow infer() method instead of infer_fast()")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
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
    
    # Choose method and output file
    if args.slow:
        method_name = "infer (SLOW)"
        output_file = "./output_v1.5_slow.mp3"
        method = tts.infer
    else:
        method_name = "infer_fast (FAST)"
        output_file = "./output_v1.5_fast.mp3"
        method = tts.infer_fast
    
    # Run inference
    print(f">> Running {method_name}...")
    start_time = time.time()
    method(
        audio_prompt="./reference.wav",
        text=test_text, 
        output_path=output_file,
        verbose=args.verbose
    )
    total_time = time.time() - start_time
    
    print(f">> {method_name} completed in {total_time:.2f} seconds")
    print(f">> Done! Check {output_file}")

if __name__ == "__main__":
    main()