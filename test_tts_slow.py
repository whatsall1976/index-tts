#!/usr/bin/env python3
"""
Test script to compare IndexTTS infer_fast vs infer methods.
Usage: python test_tts_slow.py
"""

import time
import os
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
    
    # Test text with multiple sentences
    test_text = "会自己出账单，出帐单后会给你发邮件。你付了就行了，出帐单后有宽限期，大概两周左右。也就是收到邮件的两周内付款就行了。末节决战，韩国队发起疯狂反扑，一度将分差迫近至6分。关键时刻，韩国队主力内线五犯离场成为比赛转折点，随后胡金秋在内线大杀四方，连续得分稳住局势。最终，中国男篮顶住压力，以79:71力克老对手韩国队。时隔10年之后再次杀入亚洲杯四强。"
    
    # Test 1: infer_fast method
    print("\n" + "="*50)
    print(">> Testing infer_fast method...")
    print("="*50)
    start_time = time.time()
    try:
        tts.infer_fast(
            audio_prompt="./reference.wav",
            text=test_text, 
            output_path="output_fast.mp3",
            verbose=True
        )
        fast_time = time.time() - start_time
        print(f">> infer_fast completed in {fast_time:.2f} seconds")
    except Exception as e:
        print(f">> infer_fast failed: {e}")
        fast_time = None
    
    # Test 2: infer method (original)
    print("\n" + "="*50)
    print(">> Testing infer method (original)...")
    print("="*50)
    start_time = time.time()
    try:
        tts.infer(
            audio_prompt="./reference.wav",
            text=test_text, 
            output_path="output_slow.mp3",
            verbose=True
        )
        slow_time = time.time() - start_time
        print(f">> infer completed in {slow_time:.2f} seconds")
    except Exception as e:
        print(f">> infer failed: {e}")
        slow_time = None
    
    # Comparison
    print("\n" + "="*50)
    print(">> COMPARISON RESULTS")
    print("="*50)
    if fast_time and slow_time:
        speedup = slow_time / fast_time
        print(f"infer_fast time:  {fast_time:.2f} seconds")
        print(f"infer time:       {slow_time:.2f} seconds")
        print(f"Speedup:          {speedup:.2f}x")
        
        # Check output files
        fast_files = [f for f in os.listdir('.') if f.startswith('output_fast_sentence_')]
        slow_files = [f for f in os.listdir('.') if f.startswith('output_slow')]
        
        print(f"Fast method files: {len(fast_files)} sentence files")
        print(f"Slow method files: {len(slow_files)} files")
        
        if fast_files:
            print("Fast method sentence files:")
            for f in sorted(fast_files):
                print(f"  - {f}")
        
    elif fast_time:
        print(f"infer_fast time:  {fast_time:.2f} seconds")
        print("infer method failed")
    elif slow_time:
        print(f"infer time:       {slow_time:.2f} seconds")
        print("infer_fast method failed")
    else:
        print("Both methods failed")
    
    print("\n>> Test complete!")

if __name__ == "__main__":
    main()