import os
import numpy as np

skeleton_folder = 'E:/2021/datasets/original_data/pkummd/v1/PKU_Skeleton_Renew/'
label_folder = 'E:/2021/datasets/original_data/pkummd/v1/Train_Label_PKU_final/'
save_path = 'E:/2021/datasets/original_data/pkummd/skeleton_pku_v1/'

view_dic = {'L': '001', 'M': '002', 'R': '003'}

total_file_num = 0


def array_to_skl_lines(skeleton_frame):
    skeleton_lines = ''
    for i in range(0, 75, 3):
        x = skeleton_frame[i:i + 3]
        x = map(lambda x: str(x), x)
        line = " ".join(a for a in x)
        skeleton_lines = skeleton_lines + line + '\n'
    return skeleton_lines


def transfer_and_save(file_id, file_view):
    global total_file_num
    if file_id / 10 < 1:
        file_id_ = '00' + str(file_id)
    elif file_id / 100 < 1:
        file_id_ = '0' + str(file_id)
    else:
        file_id_ = str(file_id)
    sample_file = '0' + file_id_ + '-' + file_view + '.txt'

    if not os.path.exists(label_folder + sample_file):
        print(label_folder + sample_file + ' not exist!')
        return
    else:
        print(label_folder + sample_file + ' do!')
        total_file_num += 1
    skl_list = np.loadtxt(label_folder + sample_file, delimiter=",")
    skl_file = skeleton_folder + sample_file

    skeleton_frames = np.loadtxt(skl_file)

    for i in range(0, skl_list.shape[0]):
        sample_length = int(skl_list[i][2] - skl_list[i][1]) + 1
        sample_lines = str(sample_length) + '\n'  # first lines, length of the action sample
        skeleton_sample = skeleton_frames[int(skl_list[i][1]) - 1:int(skl_list[i][2]), :150]

        for skl_id in range(0, skeleton_sample.shape[0]):
            skeleton_frame = skeleton_sample[skl_id]
            if np.sum(skeleton_frame[75:150]) == 0:
                sample_lines = sample_lines + '1\n'
                sample_lines = sample_lines + '6 1 0 0 1 1 0 -0.437266 -0.117168 2'
                sample_lines = sample_lines + '\n25\n'

                skeleton_lines = array_to_skl_lines(skeleton_frame[0:75])
                sample_lines = sample_lines + skeleton_lines
            else:
                sample_lines = sample_lines + '2\n'
                sample_lines = sample_lines + '6 1 0 0 1 1 0 -0.437266 -0.117168 2'
                sample_lines = sample_lines + '\n25\n'

                skeleton_lines_1 = array_to_skl_lines(skeleton_frame[0:75])
                skeleton_lines_2 = array_to_skl_lines(skeleton_frame[75:150])
                sample_lines = sample_lines + skeleton_lines_1 + skeleton_lines_2

        class_id = int(skl_list[i][0])
        if class_id / 10 < 1:
            class_id_ = '00' + str(class_id)
        elif class_id / 100 < 1:
            class_id_ = '0' + str(class_id)
        else:
            class_id_ = str(class_id)
        if (i + 1) / 10 < 1:
            index = '00' + str(i + 1)
        elif (i + 1) / 100 < 1:
            index = '0' + str(i + 1)
        else:
            index = str(i + 1)
        view_id = view_dic[file_view]
        save_name = 'F' + file_id_ + 'V' + view_id + 'C' + class_id_ + 'L' + index
        with open(save_path + save_name + ".skeleton", "w") as text_file:
            text_file.write(sample_lines)


for file_id in range(2, 365):
    for file_view in ['L', 'M', 'R']:
        transfer_and_save(file_id, file_view)

print('The total preprocess file:', total_file_num)
