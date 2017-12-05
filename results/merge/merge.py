def load_csv(fname, splt_char=';', skip_first_lines=False):
    song_names = list([])
    with open(fname, 'r') as fin:
        lines = fin.readlines()
        if skip_first_lines:
            lines = lines[1:]
    for line in lines:
        song_names.append(line.strip().split(splt_char)[0])
    return song_names, lines

if __name__ == '__main__':
    res_csv_name = 'result_year_fold_1_decade_cnn.csv'
    ref_csv_name = 'Year_fold_1_test.csv'    
    task = 'year'

    out_fname = ref_csv_name.replace('.csv', '_merged_results.csv')

    res_songs, res_lines = load_csv(res_csv_name, ':', skip_first_lines=True)
    ref_songs, ref_lines = load_csv(ref_csv_name, ';', skip_first_lines=True)
    
    n_ref = len(ref_songs)
    with open(out_fname, 'w') as fout:
        if task == 'year':
            fout.write('File Name:Song Title:Author:Year:Decade:Predicted\n')
        elif task == 'genre':
            fout.write('File Name:Song Title:Author:Genre:Predicted\n')
        elif task == 'gender':
            fout.write('File Name:Song Title:Author:Gender:Predicted\n')
        for i in range(n_ref):
            song_name = ref_songs[i]
            try:
                ind = res_songs.index(song_name.replace('"',''))
            except ValueError:
                print('Cannot find %s in result file' %(song_name))
                continue
            pred = res_lines[ind].strip().split(':')[1]
            if task == 'year':
                decade = int(int(ref_lines[i].strip().split(';')[-1]) / 10) * 10
                fout.write('%s:%s:%s\n' %(ref_lines[ind].strip().replace(';',':'), '%ds' %decade, pred))
            else:
                fout.write('%s:%s\n' %(ref_lines[ind].strip().replace(';',':'), pred))

