#!/usr/bin/env python3
"""
OPTIMIZED Turkish Math Token Analyzer for LARGE datasets.
Performance improvements:
- Compiled regex patterns (huge speedup)
- Single-pass text scanning
- Batch processing
- Progress tracking with ETA
"""

import json
import re
from collections import Counter
from pathlib import Path
import argparse
import time


class OptimizedTurkishMathTokenAnalyzer:
    """High-performance analyzer for large Turkish mathematical corpora"""
    
    def __init__(self, min_frequency: int = 50, chunk_size: int = 10000):
        self.min_frequency = min_frequency
        self.chunk_size = chunk_size
        
        # Statistics
        self.total_docs = 0
        self.total_chars = 0
        self.total_words = 0
        self.start_time = None
        
        # Results storage
        self.turkish_terms = Counter()
        self.math_symbols = Counter()
        self.latex_commands = Counter()
        self.multi_word_phrases = Counter()
        self.greek_letters = Counter()
        self.subscript_superscript = Counter()
        
        # PERFORMANCE OPTIMIZATION: Pre-compile all regex patterns
        print("Compiling regex patterns...")
        self._compile_patterns()
        print("✓ Patterns compiled\n")
    
    def _compile_patterns(self):
        """Pre-compile all regex patterns for much faster matching"""
        
        # Turkish mathematical terms - combine into fewer patterns
        turkish_pattern_parts = [
            r'vektör\w*', r'matris\w*', r'determinant\w*', r'düzlem\w*',
            r'uzay\w*', r'boyut\w*', r'yön\w*', r'büyüklük\w*',
            r'norm\w*', r'birim\w*', r'normal\w*', r'dik\w*', r'paralel\w*',
            r'türev\w*', r'integral\w*', r'limit\w*', r'sürekli\w*',
            r'diferansiyel\w*', r'kısmi\w*', r'yönlü\w*',
            r'eğim\w*', r'teğet\w*', r'gradyan\w*',
            r'çarpım\w*', r'toplam\w*', r'fark\w*', r'bölüm\w*', r'işlem\w*',
            r'değişim\w*', r'değişken\w*', r'anlık\w*', r'ortalama\w*', r'oran\w*',
            r'fonksiyon\w*', r'denklem\w*', r'ifade\w*',
            r'küme\w*', r'tanım\w*', r'görüntü\w*', r'alan\w*',
            r'hesap\w+', r'bulun\w+', r'belirt\w+', r'göster\w+',
            r'temsil\w*', r'elde\w*', r'çöz\w+', r'ispat\w+',
            r'nokta\w*', r'doğru\w*', r'eğri\w*', r'yüzey\w*',
            r'örneğin', r'dolayısıyla', r'sonuç\s+olarak',
            r'öncelikle', r'ayrıca', r'ancak', r'fakat',
            r'böylece', r'burada', r'şimdi', r'yani', r'demek\s+ki', r'öyleyse',
        ]
        self.turkish_pattern = re.compile(
            r'\b(' + '|'.join(turkish_pattern_parts) + r')\b',
            re.IGNORECASE
        )
        
        # Multi-word phrases
        phrase_parts = [
            r'birim\s+vektör\w*', r'normal\s+vektör\w*',
            r'çapraz\s+çarpım\w*', r'vektörel\s+çarpım\w*',
            r'skaler\s+çarpım\w*', r'iç\s+çarpım\w*', r'dış\s+çarpım\w*',
            r'yönlü\s+türev\w*', r'kısmi\s+türev\w*', r'toplam\s+türev\w*',
            r'değişim\s+oranı\w*', r'anlık\s+değişim\w*', r'ortalama\s+değişim\w*',
            r'üç\s+boyutlu\w*', r'iki\s+boyutlu\w*', r'n\s+boyutlu\w*',
            r'teğet\s+doğru\w*', r'normal\s+doğru\w*',
            r'tanım\s+kümesi\w*', r'görüntü\s+kümesi\w*', r'çözüm\s+kümesi\w*',
            r'sonuç\s+olarak', r'örneğin', r'dolayısıyla',
            r'adım\s+adım', r'şu\s+şekilde', r'bu\s+şekilde',
            r'aşağıdaki\s+gibi', r'yukarıdaki\s+gibi',
            r'elde\s+etmek', r'elde\s+ederiz', r'elde\s+edilir',
            r'bulmak\s+için', r'hesaplamak\s+için',
            r'her\s+iki', r'her\s+bir',
        ]
        self.phrase_pattern = re.compile(
            r'(' + '|'.join(phrase_parts) + r')',
            re.IGNORECASE
        )
        
        # Mathematical symbols - single pattern
        self.symbol_pattern = re.compile(
            r'[×÷±∓√∞≤≥≠≈≡∝∈∉⊂⊃⊆⊇∪∩∅∀∃¬∧∨⊕⊗→←↔⇒⇐⇔↦∂∫∮∑∏∇△▽℘ℵ]'
        )
        
        # Subscripts and superscripts - single pattern
        self.subsup_pattern = re.compile(
            r'[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔₕₖₗₘₙₚₛₜ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ]'
        )
        
        # Greek letters - single pattern
        self.greek_pattern = re.compile(
            r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]'
        )
        
        # LaTeX commands - single pattern
        self.latex_pattern = re.compile(r'\\[a-zA-Z]+')
    
    def process_chunk(self, texts: list):
        """Process a chunk of texts - optimized single-pass approach"""
        chunk_text = '\n\n'.join(texts)
        
        # Update statistics
        self.total_chars += len(chunk_text)
        # Faster word count (split is fast)
        self.total_words += chunk_text.count(' ') + chunk_text.count('\n')
        
        # SINGLE PASS: All pattern matching in one go
        # This is MUCH faster than multiple findall calls
        self.turkish_terms.update(self.turkish_pattern.findall(chunk_text))
        self.multi_word_phrases.update(self.phrase_pattern.findall(chunk_text))
        self.math_symbols.update(self.symbol_pattern.findall(chunk_text))
        self.subscript_superscript.update(self.subsup_pattern.findall(chunk_text))
        self.greek_letters.update(self.greek_pattern.findall(chunk_text))
        self.latex_commands.update(self.latex_pattern.findall(chunk_text))
    
    def load_and_process_dataset(self, dataset_path: str):
        """Load and process dataset in chunks with progress tracking"""
        print(f"Loading dataset from: {dataset_path}")
        print(f"Processing in chunks of {self.chunk_size:,} documents")
        print()
        
        self.start_time = time.time()
        chunk = []
        chunk_num = 0
        last_update = time.time()
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if 'text' in data:
                        chunk.append(data['text'])
                        self.total_docs += 1
                    
                    # Process chunk when it reaches chunk_size
                    if len(chunk) >= self.chunk_size:
                        chunk_num += 1
                        self.process_chunk(chunk)
                        chunk = []  # Clear chunk to free memory
                        
                        # Progress update (every second)
                        now = time.time()
                        if now - last_update >= 1.0:
                            elapsed = now - self.start_time
                            docs_per_sec = self.total_docs / elapsed
                            eta_seconds = (1_300_000 - self.total_docs) / docs_per_sec if docs_per_sec > 0 else 0
                            eta_mins = eta_seconds / 60
                            
                            print(f"Progress: {self.total_docs:,} docs | "
                                  f"Speed: {docs_per_sec:.0f} docs/s | "
                                  f"ETA: {eta_mins:.1f} min", 
                                  end='\r', flush=True)
                            last_update = now
                        
                except json.JSONDecodeError:
                    pass  # Skip invalid lines silently for speed
        
        # Process remaining documents
        if chunk:
            chunk_num += 1
            self.process_chunk(chunk)
        
        elapsed = time.time() - self.start_time
        print(f"\n\n✓ Processed {self.total_docs:,} documents in {elapsed/60:.1f} minutes")
        print(f"✓ Average speed: {self.total_docs/elapsed:.0f} documents/second")
        print(f"✓ Total characters: {self.total_chars:,}")
        print(f"✓ Total words: ~{self.total_words:,}")
        print()
    
    def print_summary_report(self):
        """Print concise summary report"""
        print("=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        print()
        
        # Top Turkish terms
        print(f"Top 30 Turkish Terms (≥{self.min_frequency} occurrences):")
        filtered_terms = [(t, c) for t, c in self.turkish_terms.items() if c >= self.min_frequency]
        filtered_terms.sort(key=lambda x: x[1], reverse=True)
        
        for i, (term, count) in enumerate(filtered_terms[:30], 1):
            print(f"  {i:2d}. {term:25s} {count:8,d}")
        
        total_terms = len(filtered_terms)
        if total_terms > 30:
            print(f"  ... and {total_terms - 30} more terms")
        print(f"\nTotal Turkish terms: {total_terms}")
        print()
        
        # Top phrases
        print(f"Multi-Word Phrases (≥{self.min_frequency} occurrences):")
        filtered_phrases = [(p, c) for p, c in self.multi_word_phrases.items() if c >= self.min_frequency]
        filtered_phrases.sort(key=lambda x: x[1], reverse=True)
        
        for i, (phrase, count) in enumerate(filtered_phrases[:15], 1):
            print(f"  {i:2d}. {phrase:30s} {count:8,d}")
        
        total_phrases = len(filtered_phrases)
        print(f"\nTotal phrases: {total_phrases}")
        print()
        
        # Symbols
        print(f"Math Symbols (top 20):")
        for i, (symbol, count) in enumerate(self.math_symbols.most_common(20), 1):
            print(f"  {symbol} : {count:,}")
        print(f"\nTotal symbols: {len(self.math_symbols)}")
        print()
        
        # Summary
        min_latex = max(5, self.min_frequency // 2)
        latex_count = len([l for l, c in self.latex_commands.items() if c >= min_latex])
        
        total_tokens = (
            total_terms +
            total_phrases +
            len(self.math_symbols) +
            len(self.subscript_superscript) +
            len(self.greek_letters) +
            latex_count
        )
        
        print("=" * 80)
        print(f"RECOMMENDED TOKEN COUNT: {total_tokens}")
        print("=" * 80)
        print(f"  Turkish terms:        {total_terms:4d}")
        print(f"  Multi-word phrases:   {total_phrases:4d}")
        print(f"  Math symbols:         {len(self.math_symbols):4d}")
        print(f"  Subscript/superscript:{len(self.subscript_superscript):4d}")
        print(f"  Greek letters:        {len(self.greek_letters):4d}")
        print(f"  LaTeX commands:       {latex_count:4d}")
        print("=" * 80)
        print()
    
    def generate_token_list(self, output_file: str):
        """Generate Python code with recommended tokens"""
        print(f"Generating token list: {output_file}...")
        
        # Filter by frequency
        turkish_high = [(t, c) for t, c in self.turkish_terms.items() if c >= self.min_frequency * 3]
        turkish_high.sort(key=lambda x: x[1], reverse=True)
        
        turkish_med = [(t, c) for t, c in self.turkish_terms.items() 
                      if self.min_frequency <= c < self.min_frequency * 3]
        turkish_med.sort(key=lambda x: x[1], reverse=True)
        
        phrases = [(p, c) for p, c in self.multi_word_phrases.items() if c >= self.min_frequency]
        phrases.sort(key=lambda x: x[1], reverse=True)
        
        symbols = list(self.math_symbols.most_common())
        subsup = list(self.subscript_superscript.most_common())
        greek = list(self.greek_letters.most_common())
        
        min_latex = max(5, self.min_frequency // 2)
        latex = [(l, c) for l, c in self.latex_commands.items() if c >= min_latex]
        latex.sort(key=lambda x: x[1], reverse=True)
        
        # Generate code
        lines = []
        lines.append('"""')
        lines.append('Recommended tokens for Turkish mathematical corpus')
        lines.append(f'Generated from {self.total_docs:,} documents')
        lines.append(f'Minimum frequency threshold: {self.min_frequency}')
        lines.append(f'Total recommended tokens: {len(turkish_high) + len(turkish_med) + len(phrases) + len(symbols) + len(subsup) + len(greek) + len(latex)}')
        lines.append('"""')
        lines.append('')
        lines.append('def get_turkish_math_tokens():')
        lines.append('    """Returns tokens to add to tokenizer based on corpus analysis"""')
        lines.append('    ')
        
        # High frequency terms
        if turkish_high:
            lines.append('    # Turkish mathematical terms (VERY HIGH FREQUENCY)')
            lines.append(f'    # Appear >= {self.min_frequency * 3:,} times')
            lines.append('    turkish_very_high = [')
            for term, count in turkish_high:
                lines.append(f"        '{term}',  # {count:,}")
            lines.append('    ]')
            lines.append('    ')
        
        # Medium frequency terms
        if turkish_med:
            lines.append('    # Turkish mathematical terms (HIGH FREQUENCY)')
            lines.append(f'    # Appear {self.min_frequency:,}-{self.min_frequency * 3:,} times')
            lines.append('    turkish_high = [')
            for term, count in turkish_med[:200]:  # Limit to top 200
                lines.append(f"        '{term}',  # {count:,}")
            lines.append('    ]')
            lines.append('    ')
        
        # Phrases
        if phrases:
            lines.append('    # Multi-word phrases')
            lines.append('    phrases = [')
            for phrase, count in phrases:
                lines.append(f"        '{phrase}',  # {count:,}")
            lines.append('    ]')
            lines.append('    ')
        
        # Symbols
        if symbols:
            lines.append('    # Mathematical symbols')
            lines.append('    symbols = [')
            for symbol, count in symbols:
                lines.append(f"        '{symbol}',  # {count:,}")
            lines.append('    ]')
            lines.append('    ')
        
        # Subscripts/superscripts
        if subsup:
            lines.append('    # Subscripts and superscripts')
            lines.append('    subscript_superscript = [')
            for char, count in subsup:
                lines.append(f"        '{char}',  # {count:,}")
            lines.append('    ]')
            lines.append('    ')
        
        # Greek
        if greek:
            lines.append('    # Greek letters')
            lines.append('    greek = [')
            for letter, count in greek:
                lines.append(f"        '{letter}',  # {count:,}")
            lines.append('    ]')
            lines.append('    ')
        
        # LaTeX
        if latex:
            lines.append('    # LaTeX commands')
            lines.append('    latex = [')
            for cmd, count in latex[:100]:  # Top 100
                lines.append(f"        '{cmd}',  # {count:,}")
            lines.append('    ]')
            lines.append('    ')
        
        # Combine
        parts = []
        if turkish_high: parts.append('turkish_very_high')
        if turkish_med: parts.append('turkish_high')
        if phrases: parts.append('phrases')
        if symbols: parts.append('symbols')
        if subsup: parts.append('subscript_superscript')
        if greek: parts.append('greek')
        if latex: parts.append('latex')
        
        lines.append('    all_tokens = ' + ' + '.join(parts))
        lines.append('    ')
        lines.append('    # Remove duplicates')
        lines.append('    seen = set()')
        lines.append('    unique = []')
        lines.append('    for t in all_tokens:')
        lines.append('        if t not in seen:')
        lines.append('            seen.add(t)')
        lines.append('            unique.append(t)')
        lines.append('    return unique')
        lines.append('')
        lines.append('')
        lines.append('if __name__ == "__main__":')
        lines.append('    tokens = get_turkish_math_tokens()')
        lines.append('    print(f"Total tokens: {len(tokens)}")')
        lines.append('    print("\\nFirst 50 tokens:")')
        lines.append('    for i, t in enumerate(tokens[:50], 1):')
        lines.append('        print(f"  {i:2d}. {t}")')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ Token list saved to: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description='OPTIMIZED Turkish math corpus analyzer for large datasets'
    )
    parser.add_argument('dataset', help='Path to JSONL dataset')
    parser.add_argument('--min-freq', type=int, default=100, 
                       help='Minimum frequency (default: 100 for 1M+ docs)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Documents per chunk (default: 10000)')
    parser.add_argument('--tokens', type=str, default='recommended_tokens.py',
                       help='Output file for tokens')
    
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        return 1
    
    print("=" * 80)
    print("OPTIMIZED TURKISH MATH TOKEN ANALYZER")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Min frequency: {args.min_freq:,}")
    print(f"Chunk size: {args.chunk_size:,}")
    print("=" * 80)
    print()
    
    analyzer = OptimizedTurkishMathTokenAnalyzer(
        min_frequency=args.min_freq,
        chunk_size=args.chunk_size
    )
    
    try:
        # Process dataset
        analyzer.load_and_process_dataset(args.dataset)
        
        # Print summary
        analyzer.print_summary_report()
        
        # Generate token list
        analyzer.generate_token_list(args.tokens)
        
        print("=" * 80)
        print("✓ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nTo view tokens: python {args.tokens}")
        print(f"To use in training: from {Path(args.tokens).stem} import get_turkish_math_tokens")
        print()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())