"""Download and prepare the public NHANES tabular evaluation data."""

import argparse
import bz2
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

import pandas as pd
from tqdm import tqdm

CYCLES = {'G': (2011, '2011-2012'), 'H': (2013, '2013-2014')}
PFQ_ITEMS = {
    'PFQ061B': 'difficulty_walk_quarter_mile',
    'PFQ061C': 'difficulty_walk_ten_steps',
    'PFQ061D': 'difficulty_stoop_crouch_kneel',
    'PFQ061E': 'difficulty_lift_carry',
    'PFQ061F': 'difficulty_house_chores',
    'PFQ061G': 'difficulty_prepare_meals',
    'PFQ061H': 'difficulty_walk_between_rooms',
    'PFQ061I': 'difficulty_stand_from_chair',
    'PFQ061J': 'difficulty_get_in_out_bed',
    'PFQ061K': 'difficulty_eat_utensils',
    'PFQ061L': 'difficulty_dress_self',
    'PFQ061M': 'difficulty_stand_long',
    'PFQ061N': 'difficulty_sit_long',
    'PFQ061O': 'difficulty_reach_overhead',
    'PFQ061P': 'difficulty_grasp_small_objects',
    'PFQ061Q': 'difficulty_go_out_events',
    'PFQ061R': 'difficulty_attend_social',
    'PFQ061S': 'difficulty_leisure_home',
    'PFQ061T': 'difficulty_push_pull_large',
}
TABLE_COLUMNS = {
    'DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR'],
    'BMX': ['SEQN', 'BMXBMI'],
    'WHQ': ['SEQN', 'WHD010', 'WHD020'],
    'SMQ': ['SEQN', 'SMQ020', 'SMQ040'],
    'ALQ': ['SEQN', 'ALQ101', 'ALQ120Q', 'ALQ120U'],
    'HSQ': ['SEQN', 'HSD010'],
    'PFQ': ['SEQN', 'PFQ049', 'PFQ051', 'PFQ054', 'PFQ057', 'PFQ059', *PFQ_ITEMS],
}
TABULAR_FILES_DIR = Path(__file__).resolve().parent / 'tabular_files'


def make_age(data):
    """Extract participant age."""
    return data[['RIDAGEYR']].rename(columns={'RIDAGEYR': 'age'})


def make_sex(data):
    """Extract participant sex."""
    return data['RIAGENDR'].map({1: 'Male', 2: 'Female'}).to_frame('Sex')


def make_bmi(data):
    """Use measured BMI, falling back to self-reported height and weight."""
    height = data['WHD010'].mask(data['WHD010'].isin([7777, 9999]))
    weight = data['WHD020'].mask(data['WHD020'].isin([7777, 9999]))
    reported_bmi = weight * 0.453592 / (height * 0.0254) ** 2
    return data['BMXBMI'].combine_first(reported_bmi).to_frame('BMI')


def make_smoking(data):
    """Derive current smoking from the lifetime and current-use questions."""
    smoking = pd.Series(pd.NA, index=data.index, dtype='object')
    lifetime_smoker = data['SMQ020'].eq(1)
    smoking[data['SMQ020'].eq(2) | (lifetime_smoker & data['SMQ040'].eq(3))] = 'No'
    smoking[lifetime_smoker & data['SMQ040'].isin([1, 2])] = 'Yes'
    return smoking.to_frame('Current tobacco smoking')


def make_alcohol(data):
    """Derive current and regular alcohol use."""
    frequency = data['ALQ120Q'].mask(data['ALQ120Q'].abs().lt(1e-20), 0)
    unit = data['ALQ120U'].mask(data['ALQ120U'].abs().lt(1e-20), 0)
    valid_frequency = frequency.notna() & ~frequency.isin([777, 999])

    current = pd.Series(pd.NA, index=data.index, dtype='object')
    current[frequency.isna() & data['ALQ101'].eq(2)] = 'Never'
    current[frequency.eq(0)] = 'No'
    current[valid_frequency & frequency.gt(0)] = 'Current'

    regular = pd.Series(pd.NA, index=data.index, dtype='object')
    regular[(frequency.isna() & data['ALQ101'].eq(2)) | frequency.eq(0)] = 'No'
    positive = valid_frequency & frequency.gt(0)
    regular[positive & unit.isin([1, 2])] = 'Yes'
    annual = positive & unit.eq(3)
    regular[annual & frequency.ge(12)] = 'Yes'
    regular[annual & frequency.lt(12)] = 'No'
    return pd.DataFrame({'Alcohol drinker status': current, 'Regular alcohol drinking': regular})


def make_self_health(data):
    """Label self-reported general health."""
    labels = {1: 'Excellent', 2: 'Very good', 3: 'Good', 4: 'Fair', 5: 'Poor', 7: 'Refused', 9: "Don't know"}
    return data['HSD010'].map(labels).to_frame('Overall health rating')


def make_physical_function(data):
    """Label physical-function items after applying questionnaire routing."""
    labels = {
        1: 'No difficulty',
        2: 'Some difficulty',
        3: 'Much difficulty',
        4: 'Unable to do',
        5: 'Do not do this activity',
        7: 'Refused',
        9: "Don't know",
    }
    screeners = ['PFQ049', 'PFQ051', 'PFQ054', 'PFQ057', 'PFQ059']
    no_limitation = data['RIDAGEYR'].between(20, 59) & data[screeners].eq(2).all(axis=1)
    # PFQ054 routes equipment users past the two walking items; retain the study's lower-bound recode.
    walking_equipment = data['PFQ054'].eq(1)

    output = {}
    for source, target in PFQ_ITEMS.items():
        response = data[source].mask(data[source].isna() & no_limitation, 1)
        if source in {'PFQ061B', 'PFQ061C'}:
            response = response.mask(response.isna() & walking_equipment, 2)
        output[target] = response.map(labels)
    return pd.DataFrame(output, index=data.index)


def _download(folder):
    for suffix, (year, _) in CYCLES.items():
        for table in ('PAXHD', *TABLE_COLUMNS):
            path = folder / f'{table}_{suffix}.xpt'
            if not path.exists():
                url = f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{path.name}'
                print(f'Downloading {url}')
                urlretrieve(url, path)


def _load_wearable_ids(folder):
    cache_path = folder / 'wearable_ids.csv'
    if cache_path.exists():
        return pd.read_csv(cache_path, dtype={'eid': 'int64', 'cycle': 'string', 'firmware': 'string'})

    cycles = []
    for suffix, (_, label) in CYCLES.items():
        wearable = pd.read_sas(folder / f'PAXHD_{suffix}.xpt', format='xport', encoding='utf-8')
        wearable = wearable.loc[wearable['PAXSTS'].eq(1), ['SEQN']].rename(columns={'SEQN': 'eid'})
        wearable.insert(1, 'cycle', label)
        cycles.append(wearable)
    wearable_ids = pd.concat(cycles, ignore_index=True)
    wearable_ids['eid'] = wearable_ids['eid'].astype('int64')

    # FIRMWARE_VERSION is part of each official PAX80 archive member filename:
    # https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAX80_G.htm
    # https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAX80_H.htm
    pattern = re.compile(rb'GT3XPLUS-AccelerationCalibrated-([0-9]+x[0-9]+x[0-9]+)\.')
    suffixes = {'2011-2012': 'g', '2013-2014': 'h'}

    def read_archive(record):
        eid, cycle = record
        url = f'https://ftp.cdc.gov/pub/pax_{suffixes[cycle]}/{eid}.tar.bz2'
        for byte_count in (2**17, 2**18, 2**19):
            try:
                request = Request(url, headers={'Range': f'bytes=0-{byte_count - 1}'})
                with urlopen(request, timeout=60) as response:
                    content = bz2.BZ2Decompressor().decompress(response.read(byte_count))
                match = pattern.search(content)
                if match:
                    return match.group(1).decode().replace('x', '.')
            except OSError:
                continue
        raise RuntimeError(f'Could not read firmware from {url}')

    records = list(wearable_ids[['eid', 'cycle']].itertuples(index=False, name=None))
    with ThreadPoolExecutor(max_workers=64) as executor:
        firmware = list(tqdm(executor.map(read_archive, records), total=len(records), desc='Reading firmware'))
    wearable_ids['firmware'] = pd.array(firmware, dtype='string')
    wearable_ids = wearable_ids.sort_values('eid').reset_index(drop=True)
    wearable_ids.to_csv(cache_path, index=False)
    return wearable_ids


def prepare_tabular(folder=None):
    """Download source tables and save the wearable cohort as CSV files."""
    folder = TABULAR_FILES_DIR if folder is None else Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _download(folder)
    data = _load_wearable_ids(folder)
    for name, columns in TABLE_COLUMNS.items():
        tables = []
        for suffix, (_, cycle) in CYCLES.items():
            table = pd.read_sas(folder / f'{name}_{suffix}.xpt', format='xport')[columns]
            table.insert(1, 'cycle', cycle)
            tables.append(table)
        table = pd.concat(tables, ignore_index=True).rename(columns={'SEQN': 'eid'})
        table['eid'] = table['eid'].astype('int64')
        data = data.merge(table, on=['eid', 'cycle'], how='left', validate='one_to_one')
    makers = (make_age, make_sex, make_bmi, make_smoking, make_alcohol, make_self_health, make_physical_function)
    output = pd.concat([data[['eid', 'cycle', 'firmware']], *(make(data) for make in makers)], axis=1)
    output.to_csv(folder / 'nhanes.csv', index=False)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='Download and prepare the public NHANES tabular data.')
    parser.add_argument(
        'folder',
        nargs='?',
        type=Path,
        default=TABULAR_FILES_DIR,
        help='Folder for source XPT tables and nhanes.csv (default: tabular_files beside this script).',
    )
    args = parser.parse_args(argv)
    output = prepare_tabular(args.folder)
    print(f'Saved {len(output):,} participants to {args.folder / "nhanes.csv"}')


if __name__ == '__main__':
    main()
