import sys
import cv2
import multiprocessing as mp
import feature
import utils
import stitch
import constant as const

# Parâmetros
#../image/desert/
#../image/garden/
#../image/hill/
#../image/mountain/

if __name__ == '__main__':
    input_dirname = sys.argv[1]
    pool = mp.Pool(mp.cpu_count())
    img_list, focal_length = utils.parse(input_dirname)

    cylinder_img_list = pool.starmap(utils.cylindrical_projection, [(img_list[i], focal_length[i]) for i in range(len(img_list))])

    _, img_width, _ = img_list[0].shape
    stitched_image = cylinder_img_list[0].copy()

    shifts = [[0, 0]]
    cache_feature = [[], []]

    for i in range(1, len(cylinder_img_list)):
        print('Calculando .... '+str(i+1)+'/'+str(len(cylinder_img_list)))
        img1 = cylinder_img_list[i-1]
        img2 = cylinder_img_list[i]

        print(' - Features da imagem => ', end='', flush=True)
        descriptors1, position1 = cache_feature
        if len(descriptors1) == 0:
            corner_response1 = feature.harris_corner(img1, pool)
            descriptors1, position1 = feature.extract_description(img1, corner_response1, kernel=const.DESCRIPTOR_SIZE, threshold=const.FEATURE_THRESHOLD)
        print(str(len(descriptors1))+' features encontradas.')

        print(' - Features da img_'+str(i+1)+' => ', end='', flush=True)
        corner_response2 = feature.harris_corner(img2, pool)
        descriptors2, position2 = feature.extract_description(img2, corner_response2, kernel=const.DESCRIPTOR_SIZE, threshold=const.FEATURE_THRESHOLD)
        print(str(len(descriptors2))+' features encontradas.')

        cache_feature = [descriptors2, position2]

        if const.DEBUG:
            cv2.imshow('cr1', corner_response1)
            cv2.imshow('cr2', corner_response2)
            cv2.waitKey(0)
        
        print(' - Matching => ', end='', flush=True)
        matched_pairs = feature.matching(descriptors1, descriptors2, position1, position2, pool, y_range=const.MATCHING_Y_RANGE)
        print(str(len(matched_pairs)) +' features encontradas.')

        if const.DEBUG:
            utils.matched_pairs_plot(img1, img2, matched_pairs)

        shift = stitch.RANSAC(matched_pairs, shifts[-1])
        shifts += [shift]
        print('Shift ', shift)

        print(' - Stitching => ', end='', flush=True)
        stitched_image = stitch.stitching(stitched_image, img2, shift, pool, blending=True)
        cv2.imwrite('../result/'+ str(i) +'.jpg', stitched_image)

    aligned = stitch.end2end_align(stitched_image, shifts)
    cv2.imwrite('../result/aligned.jpg', aligned)

    cropped = stitch.crop(aligned)
    cv2.imwrite('../result/result.jpg', cropped)

