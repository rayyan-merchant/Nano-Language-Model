"""Describe all fixed outputs; no editing, model-based grading or new training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
from nano_lm.artifacts import atomic_json, file_hash


def describe(root, final_grid, data, output):
    root=Path(root);protocol=json.loads((root/'configs/generation_protocol.json').read_text())
    train=(Path(data)/'train.txt').read_bytes().decode('utf-8')
    lexicon=set(re.findall(r"[a-z]+(?:'[a-z]+)?",train.lower()))
    paths={eid:root/'report/pilots'/f'{eid}_seed42/generation.json' for eid in ['E01','E02','E03','E09','E14']}
    paths['F01']=Path(final_grid);rows=[];groups=[];matched={}
    for experiment,path in paths.items():
        grid=json.loads(path.read_text(encoding='utf-8'));assert grid['protocol']==protocol and len(grid['samples'])==100
        seen=set();buckets={}
        for sample in grid['samples']:
            key=(sample['checkpoint_policy'],sample['prompt'],sample['greedy'],sample['temperature'],sample['seed'])
            assert key not in seen and sample['max_new_tokens']==256 and sample['text'].startswith(sample['prompt']);seen.add(key)
            text=sample['text'][len(sample['prompt']):];words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",text)
            interior=words[1:-1];novel=[w for w in interior if w.lower() not in lexicon]
            descriptors={}
            for size in [3,8,24,48]:
                grams=[text[i:i+size] for i in range(max(0,len(text)-size+1))]
                descriptors[f'distinct_{size}gram_fraction']=len(set(grams))/len(grams) if grams else None
                descriptors[f'repeated_{size}gram_fraction']=1-len(set(grams))/len(grams) if grams else None
            word_pairs=list(zip(words,words[1:]));pairs=len(set(word_pairs))/len(word_pairs) if word_pairs else None
            row={'experiment':experiment,'checkpoint_policy':sample['checkpoint_policy'],'prompt':sample['prompt'],
                 'greedy':sample['greedy'],'temperature':sample['temperature'],'seed':sample['seed'],
                 'checkpoint_sha256':sample['checkpoint_sha256'],'generation_file_sha256':file_hash(path),
                 'continuation_characters':len(text),'replacement_characters':text.count('\ufffd'),
                 'word_spans':len(words),'interior_train_lexicon_novel_word_spans':novel,
                 'interior_novel_fraction':len(novel)/len(interior) if interior else None,
                 'speaker_labels':re.findall(r"(?m)^([A-Za-z][A-Za-z '\-]{1,32}):\s*$",text),
                 'distinct_word_bigram_fraction':pairs,**descriptors}
            rows.append(row);group_key=(sample['checkpoint_policy'],sample['greedy'],sample['temperature'])
            buckets.setdefault(group_key,[]).append(row)
            matched.setdefault(repr(key),{})[experiment]={'original_text':sample['text'],'descriptors':row}
        for key,values in buckets.items():
            defined_pairs=[r['distinct_word_bigram_fraction'] for r in values if r['distinct_word_bigram_fraction'] is not None]
            groups.append({'experiment':experiment,'checkpoint_policy':key[0],'greedy':key[1],'temperature':key[2],
                           'outputs':len(values),'mean_continuation_characters':statistics.mean(r['continuation_characters'] for r in values),
                           'replacement_characters':sum(r['replacement_characters'] for r in values),
                           'mean_repeated_8gram_fraction':statistics.mean(r['repeated_8gram_fraction'] for r in values),
                           'mean_distinct_word_bigram_fraction':statistics.mean(defined_pairs) if defined_pairs else None,
                           'outputs_with_defined_word_bigrams':len(defined_pairs)})
    assert all(set(values)==set(paths) for values in matched.values())
    atomic_json(output,{'protocol':protocol,'selection':'All100 fixed slots for each of six genuine models; no filtering',
                'rows':rows,'groups':groups,'matched_settings':matched,
                'limitations':'String diversity/repetition, lexical novelty and speaker-label matches do not measure semantic coherence or grammar. Novel words may be real vocabulary/proper names. Boundary word spans excluded from lexical novelty. BPE and character continuations have unequal lengths at256tokens. Replacement characters are observed decoder outcomes; strict raw-byte diagnosis needs regenerated token IDs.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--generation',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();describe(a.root,a.generation,a.dataset,a.output)
