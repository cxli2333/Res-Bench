from .image_base import ImageBaseDataset
import random
from collections import Counter
import os
import re
import tempfile
from ..smp import *

class Res-Bench(ImageBaseDataset):
    TYPE = 'VQA'
    
    DATASET_URL = {
        'Res-Bench': 'https://huggingface.co/datasets/cxcx2333/ResBench/resolve/main/ResBench.tsv',
    }
    
    DATASET_MD5 = {}
    
    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        tgt_path_z = []
        if isinstance(line['image'], list):
            for i in range(len(line['image'])):
                tgt_path = osp.join(self.img_root, f"{line['index']}--{i + 1}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'][i], tgt_path)
                tgt_path_z.append(tgt_path)
        else:
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path_z.append(tgt_path)
        return tgt_path_z

    def get_spearman_corr(data):
    
        variable1 = list(map(int, data.keys())) 
        variable2 = list(data.values())         
        corr, pval = spearmanr(variable1, variable2)
        return corr

    def get_ACE(data):
        sorted_values = [data[key] for key in sorted(data, key=lambda k: int(k))]
        ACE = 0
        differences = []
        for i in range(len(sorted_values) - 1):
            diff = abs(sorted_values[i + 1] - sorted_values[i])
            ACE += diff
        return ACE

    def get_avg_and_RCE(data):
        avg_score = sum(data.values()) / len(data) if data else 0
        RCE = get_ACE(data)/avg_score if avg_score != 0 else 0
        return avg_score,RCE
        
    def data_to_list(self, data_info):
        gt = data_info['answer']
        pred = str(data_info['prediction'])
        gt = re.sub(r'[^\w\s]', '', gt).lower()
        pred = re.sub(r'[^\w\s]', '', pred).lower()
        gt_list = gt.split()
        pred_list = pred.split()
        return gt_list, pred_list

    def mathQA_judge(self,data_info):
        gt = data_info['answer'].replace(' ', '')
        pred = data_info['prediction'].replace(' ', '')
        if gt == pred:
            return True
        else:
            return False
    
    def get_ocr_grade(self, gt_list, pred_list):
        right_num = 0
        gt_counter_info = dict(Counter(gt_list))
        pdt_counter_info = dict(Counter(pred_list))
        for gt_token, gt_count in gt_counter_info.items():
            pred_count = pdt_counter_info.get(gt_token, 0)
            right_num += min(gt_count, pred_count)
        return right_num
    
    def MCQ_judge(self,data_info):
        gt = data_info['answer']
        pred = str(data_info['prediction'])
        if type(pred) != str:
            print(data_info['index'])
        gt = re.sub(r'[^\w\s]', '', gt).lower()
        pred = re.sub(r'[^\w\s]', '', pred).lower()
        if gt == pred:
            return True
        else:
            return False

    #@classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        df = pd.read_csv(eval_file, sep='\t')
    
        all_res = {'112','224','336','448','560','672','784','896','1008','1120','1232','1344'}

        l1_res_count = {}
        l1_res_grade = {}
        l1_res_score = {}

        l2_res_count = {}
        l2_res_grade = {}
        l2_res_score = {}
        
        l1_avg_score = {}
        l2_avg_score = {}

        l1_spearman_corr = {}
        l2_spearman_corr = {}

        l1_ACE={}
        l2_ACE={}

        l1_RCE={}
        l2_RCE={}
        overall_score = {}
        overall_avg_score = 0
        overall_spearman_corr = 0
        overall_ACE = 0
        overall_RCE = 0

        dict_list = df.to_dict(orient='records')
        
        for data_info in dict_list:
            l1_category = data_info['l1-category']
            l2_category = data_info['l2-category']  
            res = str(data_info['resolution'])
            if l1_category not in l1_res_count:
                l1_res_count[l1_category] = {r: 0 for r in all_res}
                l1_res_grade[l1_category] = {r: 0 for r in all_res}

            if l2_category not in l2_res_count:
                l2_res_count[l2_category] = {r: 0 for r in all_res}
                l2_res_grade[l2_category] = {r: 0 for r in all_res}

            if data_info['question_type'] == "MCQ":
                l1_res_count[l1_category][res] += 1
                l2_res_count[l2_category][res] += 1
                if MCQ_judge(data_info):
                    l1_res_grade[l1_category][res] += 1
                    l2_res_grade[l2_category][res] += 1

            elif data_info['question_type'] == "VQA":
                l2_res_count[l2_category][res] += 1
                l1_res_count[l1_category][res] += 1
                gt_list, pred_list = data_to_list(data_info)
                right_num = get_ocr_grade(gt_list, pred_list)
                score = right_num / len(gt_list) if len(gt_list) > 0 else 0
                l2_res_grade[l2_category][res] += score
                l1_res_grade[l1_category][res] += score
            
            elif data_info['question_type'] == "mathQA":
                l2_res_count[l2_category][res] += 1
                l1_res_count[l1_category][res] += 1
                if mathQA_judge(data_info):
                    l2_res_grade[l2_category][res] += 1
                    l1_res_grade[l1_category][res] += 1

        for l2_cate in l2_res_grade:
            l2_res_score[l2_cate] = {}
            for res in l2_res_grade[l2_cate]:
                count = l2_res_count[l2_cate][res]
                grade = l2_res_grade[l2_cate][res]
                l2_res_score[l2_cate][res] = grade / count if count > 0 else 0

            #l2_res_score[l2_cate]['avg'] = sum(l2_res_score[l2_cate].values()) / len(l2_res_score[l2_cate]) if l2_res_score[l2_cate] else 0
            l2_spearman_corr[l2_cate] = get_spearman_corr(l2_res_score[l2_cate])
            l2_ACE[l2_cate] = get_ACE(l2_res_score[l2_cate])
            l2_avg_score[l2_cate],l2_RCE[l2_cate] = get_avg_and_RCE(l2_res_score[l2_cate])

        for l1_cate in l1_res_grade:
            l1_res_score[l1_cate] = {}
            for res in l1_res_grade[l1_cate]:
                count = l1_res_count[l1_cate][res]
                grade = l1_res_grade[l1_cate][res]
                l1_res_score[l1_cate][res] = grade / count if count > 0 else 0

            #l1_res_score[l1_cate]['avg'] = sum(l1_res_score[l1_cate].values()) / len(l1_res_score[l1_cate]) if l1_res_score[l1_cate] else 0
            l1_spearman_corr[l1_cate] = get_spearman_corr(l1_res_score[l1_cate])
            l1_ACE[l1_cate] = get_ACE(l1_res_score[l1_cate])
            l1_avg_score[l1_cate],l1_RCE[l1_cate] = get_avg_and_RCE(l1_res_score[l1_cate])

        for key1 in l1_res_score:
            for key2 in l1_res_score[key1]:
                if key2 not in overall_score:
                    overall_score[key2] = 0
                overall_score[key2] += l1_res_score[key1][key2] / 6
        overall_spearman_corr = get_spearman_corr(overall_score)
        overall_ACE = get_ACE(overall_score)
        overall_avg_score,overall_RCE = get_avg_and_RCE(overall_score)

        all_results = {
            'overall_score': overall_score,
            'overall_avg_score': overall_avg_score,
            'overall_spearman_corr': overall_spearman_corr,
            'overall_ACE': overall_ACE,
            'overall_RCE': overall_RCE,
            'l1_res_score': l1_res_score,
            'l2_res_score': l2_res_score,
            'l1_avg_score': l1_avg_score,
            'l2_avg_score': l2_avg_score,
            'l1_spearman_corr': l1_spearman_corr,
            'l2_spearman_corr': l2_spearman_corr,
            'l1_ACE': l1_ACE,
            'l2_ACE': l2_ACE,
            'l1_RCE': l1_RCE,
            'l2_RCE': l2_RCE
        }


        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(all_results, score_pth)
        return all_results
        
    def build_prompt(self, line):
    
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        text = line['question']
        
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        
        msgs.append(dict(type='text', value=text))
        return msgs
    
