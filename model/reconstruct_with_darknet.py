import os
from tqdm import tqdm

from model.darknet import detect, load_net, load_meta

gpu_index = 0



DATA_DIR = "../data/preprocessed_input"

train_dcm_dir = os.path.join(DATA_DIR, "stage_1_train_images")
test_dcm_dir = os.path.join(DATA_DIR, "stage_1_test_images")

img_dir = os.path.join(os.getcwd(), "images")  # .jpg
label_dir = os.path.join(os.getcwd(), "labels")  # .txt
metadata_dir = os.path.join(os.getcwd(), "metadata") # .txt

# YOLOv3 config file directory
cfg_dir = os.path.join(os.getcwd(), "cfg")
# YOLOv3 training checkpoints will be saved here
backup_dir = os.path.join(os.getcwd(), "backup")


submit_file_path = "submission.csv"
cfg_path = os.path.join(cfg_dir, "rsna_yolov3.cfg_test")
weight_path = os.path.join(backup_dir, "rsna_yolov3_15300.weights")

test_img_list_path = os.path.join(metadata_dir, "te_list.txt")


net = load_net(cfg_path.encode(),
               weight_path.encode(),
               gpu_index)
meta = load_meta(data_extention_file_path.encode())



def reconstruct_with_darknet():
    submit_dict = {"patientId": [], "PredictionString": []}

    with open(test_img_list_path, "r") as test_img_list_f:
        # tqdm run up to 1000(The # of test set)
        for line in tqdm(test_img_list_f):
            patient_id = line.strip().split('/')[-1].strip().split('.')[0]

            infer_result = detect(net, meta, line.strip().encode(), thresh=0.2)

            submit_line = ""
            for e in infer_result:
                confi = e[1]
                w = e[2][2]
                h = e[2][3]
                x = e[2][0] - w / 2
                y = e[2][1] - h / 2
                submit_line += "{} {} {} {} {} ".format(confi, x, y, w, h)

            submit_dict["patientId"].append(patient_id)
            submit_dict["PredictionString"].append(submit_line)

    pd.DataFrame(submit_dict).to_csv(submit_file_path, index=False)