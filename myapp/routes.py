from myapp import application
from flask import render_template #, redirect, url_for, send_from_directory
from myapp.forms import TextForm
#from werkzeug.utils import secure_filename
import os
from os.path import join, dirname, abspath #, exists
import sys # for debugging
import numpy as np

from myapp.main import cap_img_partial

from myapp.main import cap_vid_partial

from os import listdir
from datetime import datetime

@application.route('/', methods=['GET', 'POST'])
@application.route('/index', methods=['GET', 'POST'])
def index():
    form = TextForm()
    if form.validate_on_submit():
        # print(form.text.data, file=sys.stdout)
        if form.text.data == "asd":
            # just a magic test for the image display
            APP_ROOT = dirname(abspath(__file__))
            EXAMPLE_IMAGE_FOLDER = join(APP_ROOT, "static/example_images")
            filepaths_upload = listdir(EXAMPLE_IMAGE_FOLDER)
            random_int = np.random.randint(len(filepaths_upload))
            img = filepaths_upload[random_int]
            # print(img, file=sys.stdout)
            generated_img_fp = join(EXAMPLE_IMAGE_FOLDER, img) #join(EXAMPLE_IMAGE_FOLDER, img)
            # print(generated_img_fp, file=sys.stdout)
        else:
            user_caption = form.text.data
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(" ", "_").replace(':', '_').replace('-', '_')
            filename_img = 'image_{0}_{1}.png'.format(user_caption.replace(" ", "_"), timestamp)
            filename_vid = 'video_{0}_{1}.mp4'.format(user_caption.replace(" ", "_"), timestamp)
            filename_vid_new = 'video_{0}_{1}_{2}.mp4'.format(user_caption.replace(" ", "_"), timestamp,'new')
            APP_ROOT = dirname(abspath(__file__))
            output_dir = join('static', "outputs")
            generated_img_fp = join(output_dir, filename_img)
            generated_vid_fp = join(output_dir, filename_vid)
            generated_vid_new = join(output_dir, filename_vid_new)
            # _, _, caption, main_ls = cap_img_partial(user_caption, file_path=generated_img_fp)
            
            # video is saved in the function
            # main_ls is the main label of the caption
            _, _ = cap_vid_partial(cap=user_caption, file_path_img=generated_img_fp, file_path_vid=generated_vid_fp)

            # workaround for ubuntu web browser codecs
            os.system(f"ffmpeg -i {generated_vid_fp} -vcodec libx264 {generated_vid_new}")
            


        return render_template('mat_index.html', form=form ,
                               generated_img_fp=generated_img_fp,
                               caption=user_caption,
                               generated_vid_fp=generated_vid_new)

    return render_template('mat_index.html', form=form)

# @application.route('/', methods=['GET', 'POST'])
# @application.route('/index', methods=['GET', 'POST'])
# def index():
#     form = ImageForm()
#
#     if form.validate_on_submit():
#
#         APP_ROOT = dirname(abspath(__file__))
#         DEFAULT_IMAGE_FOLDER = join(APP_ROOT, "static/image")
#         UPLOAD_FOLDER = join(APP_ROOT, 'static/uploads')
#         PROCESSED_IMAGE_FOLDER = join(APP_ROOT, "static/processed_image")
#
#         # LUT file handling
#         LUT = form.your_cube_lut.data
#         lut_filename = secure_filename(LUT.filename)
#         uploaded_LUT_filepath = join(UPLOAD_FOLDER, lut_filename)
#         LUT.save(uploaded_LUT_filepath)
#
#         # image file handling
#         image = form.your_image.data
#         # case no image was uploaded, use default neutral-lut.png instead
#         if image == None:
#             #
#             # print("TESTING NO IMAGE", file=sys.stdout)
#             # print(image, file=sys.stdout)
#             # print(image == None, file=sys.stdout)
#             # use static/image/neutral-lut.png
#             image_filename = "neutral-lut.png"
#             uploaded_image_filepath = join(DEFAULT_IMAGE_FOLDER, image_filename)
#             up_im_fp = join('static/image', image_filename)
#         else:
#
#             # print("TESTING WITH IMAGE", file=sys.stdout)
#             # print(image, file=sys.stdout)
#             # print(image == None, file=sys.stdout)
#             image_filename = secure_filename(image.filename)  #
#             uploaded_image_filepath = join(UPLOAD_FOLDER, image_filename)  #
#             image.save(uploaded_image_filepath)  #
#             up_im_fp = join('static/uploads', image_filename)  #
#
#         processed_image_filename = lut_filename[:-5] + "_processed_" + image_filename  #
#         processed_image_filepath = join(PROCESSED_IMAGE_FOLDER, processed_image_filename)
#         # lut_fp = join('static/uploads', lut_filename)
#         pr_im_fp = join('static/processed_image', processed_image_filename)
#
#         webapp_LUT_processing(image_filepath=uploaded_image_filepath, LUT_filepath=uploaded_LUT_filepath,
#                               output_filepath=processed_image_filepath)
#
#         # feeds template with image filepaths for display and name for download Button
#         return render_template('index.html', form=form, uploaded_image_filepath=up_im_fp,
#                                processed_image_filepath=pr_im_fp, processed_image_filename=processed_image_filename)
#
#     return render_template('index.html', form=form)
#
#
# @application.route('/download/')
# @application.route('/download/<filename>', methods=['GET'])
# def downloadFile(filename=None):
#     ''''Downloads the image from folder safely,
#         Request Sent by Download Button'''
#
#     # catch case where no image was processed
#     if filename == None:
#         return redirect(url_for("index"))
#
#     path = join("static", "processed_image")
#     APP_ROOT_2 = dirname(abspath(__file__))
#     PROCESSED_IMAGE_FOLDER = join(APP_ROOT_2, "static/processed_image", filename)
#
#     # catch case where image does not exist / prevent url injection
#     if exists(PROCESSED_IMAGE_FOLDER):
#         # print("TEST SUCCESS", file=sys.stdout)
#         return send_from_directory(path, filename, as_attachment=True)
#
#     else:
#         # print("TEST DOES NOT EXIST", file=sys.stdout)
#         return redirect(url_for("index"))
#
#
# @application.route("/delete_files")
# def delete_files():
#     '''Deletes all the uploaded and processed files from static directories'''
#     APP_ROOT = dirname(abspath(__file__))
#     UPLOAD_FOLDER = join(APP_ROOT, 'static/uploads')
#     PROCESSED_IMAGE_FOLDER = join(APP_ROOT, "static/processed_image")
#
#     filepaths_upload = listdir(UPLOAD_FOLDER)
#     filepaths_processed = listdir(PROCESSED_IMAGE_FOLDER)
#
#     # delete files if they exist, if not present creates infinity loop
#     if filepaths_upload:
#         for fp in filepaths_upload:
#             remove(join(UPLOAD_FOLDER, fp))
#     if filepaths_processed:
#         for fp in filepaths_processed:
#             remove(join(PROCESSED_IMAGE_FOLDER, fp))
#
#     return redirect(url_for("index"))
