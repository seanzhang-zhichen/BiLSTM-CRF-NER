
from .help import flatten_lists

def _find_tag(labels, B_label="B-COM",I_label="M-COM", E_label="E-COM", S_label="S-COM"):
    result = []
    lenth = 0
    for num in range(len(labels)):
        if labels[num] == B_label:
            song_pos0 = num
        if labels[num] == B_label and labels[num+1] == E_label:
            lenth = 2
            result.append((song_pos0,lenth))

        if labels[num] == I_label and labels[num-1] == B_label:
            lenth = 2
            for num2 in range(num,len(labels)):
                if labels[num2] == I_label and labels[num2-1] == I_label:
                    lenth += 1
                if labels[num2] == E_label:
                    lenth += 1
                    result.append((song_pos0,lenth))
                    break
        if labels[num] == S_label:
            lenth = 1
            song_pos0 = num
            result.append((song_pos0,lenth))
            
    return result


tags = [("B-NAME","M-NAME", "E-NAME", "S-NAME"),
        ("B-TITLE","M-TITLE", "E-TITLE", "S-TITLE"), 
        ("B-ORG","M-ORG", "E-ORG", "S-ORG"), 
        ("B-RACE","M-RACE", "E-RACE", "S-RACE"), 
        ("B-EDU","M-EDU", "E-EDU", "S-EDU"), 
        ("B-CONT","M-CONT", "E-CONT", "S-CONT"),
        ("B-LOC","M-LOC", "E-LOC", "S-LOC"), 
        ("B-PRO","M-PRO", "E-PRO", "S-PRO")]

 
def find_all_tag(labels):
    result = {}
    for tag in tags:
        res = _find_tag(labels, B_label=tag[0], I_label=tag[1], E_label=tag[2], S_label=tag[3])
        result[tag[0].split("-")[1]] = res
    return result

def precision(pre_labels,true_labels):
    '''
    :param pre_tags: list
    :param true_tags: list
    :return:
    '''
    pre = []
    pre_labels = flatten_lists(pre_labels)
    true_labels = flatten_lists(true_labels)

    pre_result = find_all_tag(pre_labels)
    true_result = find_all_tag(true_labels)

    result_dic = {}
    for name in pre_result:
        for x in pre_result[name]:
            if result_dic.get(name) is None:
                result_dic[name] = []
            if x:
                if pre_labels[x[0]:x[0]+x[1]] == true_labels[x[0]:x[0]+x[1]]:
                    result_dic[name].append(1)
                else:
                    result_dic[name].append(0)
        # print(f'tag: {name} , length: {len(result_dic[name])}')
    
    sum_result = 0
    for name in result_dic:
        sum_result += sum(result_dic[name])
        # print(f'tag2: {name} , length2: {len(result_dic[name])}')
        result_dic[name] = sum(result_dic[name]) / len(result_dic[name])

    for name in pre_result:
        for x in pre_result[name]:
            if x:
                if pre_labels[x[0]:x[0]+x[1]] == true_labels[x[0]:x[0]+x[1]]:
                    pre.append(1)
                else:
                    pre.append(0)
    total_precision = sum(pre)/len(pre)

    return total_precision, result_dic