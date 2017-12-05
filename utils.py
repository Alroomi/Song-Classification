def parse_csv_list_file(list_fname):
    cls_to_id = dict({})
    id_to_cls = dict({})
    all_labels = list([])
    samples = list([])
    with open(list_fname, 'r') as fin:
        lines = fin.readlines()
        n_lines = len(lines)
        for i in range(1, n_lines): #first line contains only headers
            splt = lines[i].strip().split(';')
            fname = splt[0]
            label = splt[-1].lower()
            samples.append([fname.replace('"',''), label])
            all_labels.append(label)
        all_labels = sorted(all_labels)
        label_counter = 0
        for label in all_labels:
            if label not in cls_to_id:
                cls_to_id[label] = label_counter
                id_to_cls[label_counter] = label
                label_counter += 1
        for sample in samples:
            sample[1] = cls_to_id[sample[1]]
    return samples, cls_to_id, id_to_cls
