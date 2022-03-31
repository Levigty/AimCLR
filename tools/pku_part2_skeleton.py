import os
import numpy as np

skeleton_folder = 'E:/2021/datasets/original_data/pkummd/v2/skeleton/'
label_folder = 'E:/2021/datasets/original_data/pkummd/v2/label/'
save_path = 'E:/2021/datasets/original_data/pkummd/skeleton_pku_v2/'

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


def transfer_and_save(a_id, n_id, file_view):
    global total_file_num
    if a_id < 10:
        a_id = '0' + str(a_id)
    else:
        a_id = str(a_id)
    if n_id < 10:
        n_id = '0' + str(n_id)
    else:
        n_id = str(n_id)
    sample_file = 'A' + a_id + 'N' + n_id + '-' + file_view + '.txt'

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
        save_name = 'A0' + a_id + 'N0' + n_id + 'V' + view_id + 'C' + class_id_ + 'L' + index
        with open(save_path + save_name + ".skeleton", "w") as text_file:
            text_file.write(sample_lines)


for a_id in range(1, 27):
    for n_id in range(1, 14):
        for file_view in ['L', 'M', 'R']:
            transfer_and_save(a_id, n_id, file_view)

print('The total preprocess file:', total_file_num)
