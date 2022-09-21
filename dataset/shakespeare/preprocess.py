"""Preprocesses the Shakespeare dataset for federated training.
Copyright 2017 Google Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
To run:
  python preprocess_shakespeare.py path/to/raw/shakespeare.txt output_directory/
The raw data can be downloaded from:
  http://www.gutenberg.org/cache/epub/100/pg100.txt
(The Plain Text UTF-8 file format, md5sum: 036d0f9cf7296f41165c2e6da1e52a0e)
Note that The Comedy of Errors has a incorrect indentation compared to all the
other plays in the file. The code below reflects that issue. To make the code
cleaner, you could fix the indentation in the raw shakespeare file and remove
the special casing for that play in the code below.
Authors: loeki@google.com, mcmahan@google.com
Disclaimer: This is not an official Google product.
"""
import re

import numpy as np
import torch
import collections
from pathlib import Path

# tools
from sklearn.model_selection import train_test_split


def match_character(line, has_error=False):
    if has_error:
        return re.compile(r'^([a-zA-Z][a-zA-Z ]*)\. (.*)').match(line)
    else:
        return re.compile(r'^  ([a-zA-Z][a-zA-Z ]*)\. (.*)').match(line)


def match_continuation(line, has_error=False):
    if has_error:
        return re.compile(r'^(.*)').match(line)
    else:
        return re.compile(r'^    (.*)').match(line)


def split_sentence(text: str, seq_length: int = 80):
    text = re.sub(r"   *", r' ', text.replace('\n', ' '))
    data, target = [], []
    for i in range(0, len(text) - seq_length):
        data.append(text[i:i + seq_length])
        target.append(text[i + seq_length])
    return data, target


# step1
def split_by_play(datasource):
    """Splits the full data by play."""
    # List of tuples (play_name, dict from character to list of lines)
    # Track discarded lines.
    plays, discarded_lines = [], []
    lines = datasource.splitlines(True)[1:]
    # skip contents, the sonnets, and all's well that ends well
    found, start_idx = False, 0
    for i, line in enumerate(lines):
        if 'by William Shakespeare' not in line:
            continue
        elif not found:
            found = True
        else:
            start_idx = i - 5
            lines = lines[start_idx:]
            break
    current_character = None
    comedy_of_errors = False
    for i, line in enumerate(lines):
        # This marks the end of the plays in the file.
        if i > 124195 - start_idx:
            break
        # This is a pretty good heuristic for detecting the start of a new play:
        if 'by William Shakespeare' in line:
            current_character = None
            characters = collections.defaultdict(list)
            # The title will be 2, 3, 4, 5, 6, or 7 lines above "by William Shakespeare".
            if lines[i - 2].strip():
                title = lines[i - 2]
            elif lines[i - 3].strip():
                title = lines[i - 3]
            elif lines[i - 4].strip():
                title = lines[i - 4]
            elif lines[i - 5].strip():
                title = lines[i - 5]
            elif lines[i - 6].strip():
                title = lines[i - 6]
            else:
                title = lines[i - 7]
            title = title.strip()

            assert title, f'Parsing error on line {i}. Expecting title 2 or 3 lines above.'
            comedy_of_errors = (title == 'THE COMEDY OF ERRORS')
            # Degenerate plays are removed at the end of the method.
            plays.append((title, characters))
            continue
        if tmp := match_character(line, comedy_of_errors):
            character, snippet = tmp.group(1), tmp.group(2)
            # Some character names are written with multiple casings, e.g., SIR_Toby
            # and SIR_TOBY. To normalize the character names, we uppercase each name.
            # Note that this was not done in the original preprocessing and is a
            # recent fix.
            character = character.upper()
            if not (comedy_of_errors and character.startswith('ACT ')):
                characters[character].append(snippet)
                current_character = character
                continue
            else:
                current_character = None
                continue
        elif current_character:
            if tmp := match_continuation(line, comedy_of_errors):
                if comedy_of_errors and tmp.group(1).startswith('<'):
                    current_character = None
                    continue
                else:
                    characters[current_character].append(tmp.group(1))
                    continue
        # Didn't consume the line.
        line = line.strip()
        if line and i > 2646:
            # Before 2646 are the sonnets, which we expect to discard.
            discarded_lines.append('%d:%s' % (i, line))
    # Remove degenerate "plays".
    return [play for play in plays if len(play[1]) > 1], discarded_lines


# step2
def group_by_user(plays, output: Path):
    def to_play_character(play_, character_):
        return re.sub('\\W+', '_', (play_ + '_' + character_).replace(' ', '_'))

    """
      Splits character data into train and test sets.
      if test_fraction <= 0, returns {} for all_test_examples
      plays := list of (play, dict) tuples where play is a string and dict
      is a dictionary with character names as keys
    """
    skipped_characters = 0
    data = collections.defaultdict(list)
    user_plays = {}
    for play, characters in plays:
        curr_characters = list(characters.keys())
        for c in curr_characters:
            user_plays[to_play_character(play, c)] = play
        for character, sound_bites in characters.items():
            samples = [(play, character, sound_bite) for sound_bite in sound_bites]
            if len(samples) <= 2:
                # Skip characters with fewer than 2 lines since we need at least one
                # train and one test line.
                skipped_characters += 1
                continue
            for pl, ch, sb in samples:
                data[to_play_character(pl, ch)].append(sb)
    # 写入文件
    torch.save(user_plays, output.joinpath('user_plays.pt'))
    for character_name, sound_bites in data.items():
        with open(output.joinpath(f'{character_name}.txt'), 'w') as f:
            for sound_bite in sound_bites:
                f.write(sound_bite + '\n')


# step3
def to_datasource(path: Path, test_ratio: float):
    user_plays = torch.load(path.joinpath('user_plays.pt'))
    users, hierarchies, num_samples, user_data = [], [], {}, {}
    for f in filter(lambda x: str(x).endswith(".txt"), path.iterdir()):
        user = f.parts[-1][:-4]
        with open(f, 'r') as inf:
            data, target = split_sentence(inf.read())
            if data:
                users.append(user)
                num_samples[user] = len(target)
                user_data[user] = str(path.joinpath(f'{user}.pt'))
                train, test = train_test_split(np.array(list(zip(data, target))), test_size=test_ratio)
                torch.save({'train': train, 'test': test}, user_data[user])
                hierarchies.append(user_plays[user])
    torch.save(
        {'users': users, 'hierarchies': hierarchies, 'num_samples': num_samples, 'user_data': user_data},
        path.joinpath('index.pt')
    )


def preprocessing(root: str, test_ratio: float = 0.2):
    raw = Path(f'{root}/pg100.txt')
    processed = Path(f'{root}/processed/')
    if not processed.exists():
        processed.mkdir()
    with open(raw, 'r') as ds:
        plays, _ = split_by_play(ds.read())
        group_by_user(plays, processed)
        to_datasource(processed, test_ratio)
