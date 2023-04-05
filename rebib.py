import pybtex
from pybtex.database.input import bibtex
import bibtex_dblp.dblp_api
import requests
import numpy as np
import multiprocessing as mp
from absl import flags
from absl import app
import sys
import os
from tqdm.auto import tqdm


flags.DEFINE_bool('reformat_only', False, 'only reformat the bib file')
flags.DEFINE_string('bibfile', 'ref.bib', 'the .bib file to process')
flags.DEFINE_integer('num_workers', 5, 'parallelization')
flags.DEFINE_bool('interactive', True, 'manually choose a candidate when there are two candidates')
flags.DEFINE_string('format', 'bibtex', 'format of the file')
flags.DEFINE_bool('allow_duplicate', False, 'allow duplicated bib entry')
flags.DEFINE_bool('verbose', False, 'verbose')
flags.FLAGS(sys.argv)
FLAGS = flags.FLAGS

def filter_fields(pub):
    desired = ['title', 'booktitle', 'year', 'journal', 'school']
    return {k: pub.fields[k] for k in desired if k in pub.fields.keys()}


def update_entry_with_pub(entry, pub):
    bytes = requests.get(f'{pub.url}.bib')
    updated_entries = pybtex.database.parse_bytes(bytes.content, bib_format=FLAGS.format)
    assert len(updated_entries.entries) == 1
    for new_key in updated_entries.entries:
        updated_entry = updated_entries.entries[new_key]
    if pub.venue == 'CoRR':
        # hard-coding for arxiv publications
        updated_entry.fields['journal'] = f'arXiv preprint arXiv:{updated_entry.fields["volume"][4:]}'
    updated_entry = pybtex.database.Entry(
        type_=updated_entry.type,
        persons=updated_entry.persons,
        fields=updated_entry.fields
    )
    if 'editor' in updated_entry.persons.keys():
        del updated_entry.persons['editor']
    updated_entry.key = entry.key
    return updated_entry


def update_entry_wrapper(entry):
    result = None
    for _ in range(5):
        try:
            result = update_entry(entry)
            break
        except requests.RequestException as e:
            print(f'Error: {e} when processing {entry_to_str(entry)}')
            continue
        except Exception:
            raise
    result = result or dict(succeeded=None, failed=entry, info=None, pending=None, skipped=None, original=entry)
    return result


def pub_to_str(pub):
    authors = ' '.join([author.name['text'] for author in pub.authors])
    return f'{authors}: {pub.title} @ {pub.venue}, {pub.year}'


def entry_to_str(entry, short=False):
    if not short:
        other_fields = ",\n".join([f"{k}={v}" for k, v in entry.fields.items() if k not in ["author", "title", "journal", "year"]])
    else:
        other_fields =  None
    authors = ', '.join(map(str, entry.persons['author']))
    return (f'{authors}: {entry.fields["title"]} @ {entry.fields.get("journal", "--")}, {entry.fields.get("year", "--")}'
            # all other fields, except author, title, journal, year, joined by ', '
            + (f' & other fields:\n{other_fields}\n)' if other_fields else ''))


def update_entry(entry):
    author = str(entry.persons['author'][0] if 'author' in entry.persons.keys() else '')
    title = entry.fields['title']
    query = f'{title} {author}'
    result = dict(original=entry, succeeded=None, failed=None, info=None, pending=None, skipped=None)

    # if the entry has bibsource and biburl, skip it
    if 'bibsource' in entry.fields.keys() and 'biburl' in entry.fields.keys():
        result['skipped'] = entry
        return result
    
    # Otherwise, try to update it.
    try:
        search_results = bibtex_dblp.dblp_api.search_publication(query, max_search_results=2)
    except:
        result['failed'] = entry
        result['info'] = 'DBLP request error'
        return result
    if search_results.total_matches == 0:
        result['failed'] = entry
    else:
        pubs = [result.publication for result in search_results.results]
        is_arxiv = [pub.venue == 'CoRR' for pub in pubs]
        if len(is_arxiv) == 1:
            # only one match, no other choice
            pub = pubs[0]
        elif np.sum(is_arxiv) == 1:
            # one is arxiv and the other is not, use the non-arxiv one
            pub = pubs[np.argmin(is_arxiv)]
        else:
            result['pending'] = (query, pubs, entry)
            return result
        result['succeeded'] = update_entry_with_pub(entry, pub)

    return result


def rebib(argv):
    """
    Rebib: a tool to update the bib entries in a .bib file.
    """
    if FLAGS.allow_duplicate:
        import pybtex.errors
        pybtex.errors.set_strict_mode(False)

    parser = bibtex.Parser(encoding="UTF-8")
    bib_data = parser.parse_file(FLAGS.bibfile)
    entries = [bib_data.entries[key] for key in bib_data.entries]

    print()
    if not FLAGS.reformat_only:
        if FLAGS.num_workers > 1:
            pool = mp.Pool(processes=FLAGS.num_workers)
            results = pool.map(update_entry_wrapper, entries)
        else:
            results = [update_entry_wrapper(entry) for entry in tqdm(entries)]

        for res in results:
            if res['pending'] is not None:
                query, pubs, entry = res['pending']
                if FLAGS.interactive:
                    hint = (f'Select a candidate for {query}, i.e.:\n{entry_to_str(entry)}\n\n (0) Skip\n '
                        + ''.join(f' ({i + 1}) {pub_to_str(pub)}\n' for i, pub in enumerate(pubs))
                    )
                    selected = int(input(hint))
                    if not selected:
                        res['skipped'] = entry
                    else:
                        pub = pubs[selected - 1]
                        res['succeeded'] = update_entry_with_pub(entry, pub)
                else:
                    res['skipped'] = entry

        updated = []
        untouched = []
        skipped = []
        for res in results:
            if res['skipped'] is not None:
                skipped.append(res['skipped'])
            if res['succeeded'] is not None:
                updated.append(res['succeeded'])
            if res['failed'] is not None:
                untouched.append(res['failed'])
            if res['info'] is not None:
                print(res['info'])

        for prefix, entries in [('skipped', skipped), ('updated', updated), ('failed', untouched)]:
            for entry in entries:
                print(prefix, entry_to_str(entry, short=True))
    else:
        results = [dict(original=entry, succeeded=None) for entry in entries]
            
    dir, file = os.path.split(FLAGS.bibfile)
    file, _ = os.path.splitext(file)

    # overwrite original bibtex file (this is okay when we are using git)
    with open(os.path.join(dir, f'{file}.bib'), 'w') as f:
        for res in results:
            # write the succeeded entry or the orignal entry (otherwise)
            entry = res['succeeded'] or res['original']
            # unescape the fields of the entry
            for k, v in entry.fields.items():
                # \& -> &
                entry.fields[k] = v.replace('\\\\', '\\')

            f.write(entry.to_string(FLAGS.format, encoding="UTF-8"))
            f.write('\n')

    # # write out updated entries
    # with open(os.path.join(dir, f'rebib_{file}_updated.bib'), 'w') as f:
    #     for entry in updated:
    #         f.write(entry)
    #         f.write('\n')
    # # write out skipped entries
    # with open(os.path.join(dir, f'rebib_{file}_skipped.bib'), 'w') as f:
    #     for entry in skipped:
    #         f.write(entry)
    #         f.write('\n')
    # # write out ountouced entries
    # with open(os.path.join(dir, f'rebib_{file}_untouched.bib'), 'w') as f:
    #     for entry in untouched:
    #         f.write(entry)
    #         f.write('\n')


if __name__ == '__main__':
    app.run(rebib)
