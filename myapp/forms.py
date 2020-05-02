from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, ValidationError, TextField
from wtforms.validators import Optional, Required, Length
from flask_wtf.file import FileField, FileRequired, FileAllowed

# custom filename length validator to keep filename in a range
# def filename_length(min=-1, max=-1):
#     message = 'Must be between %d and %d characters long.' % (min, max)
#
#     def _length(form, field):
#         l = field.data.filename and len(field.data.filename) or 0
#         if l < min or max != -1 and l > max:
#             raise ValidationError(message)
#
#     return _length
#
# class ImageForm(FlaskForm):
#     your_image = FileField("image", validators=[Optional(), filename_length(1,128), FileAllowed(['jpg', 'png'], 'Image files only!')])
#     your_cube_lut = FileField(".cube LUT", validators=[FileRequired(), filename_length(1,128), FileAllowed(["cube"],".cube files only")])
#     submit = SubmitField('apply LUT to image')

# include regex validator to filter out non TEXT STUFF
class TextForm(FlaskForm):
    text = TextField("Describe what you would like to see", [Required(), Length(1,128,"input too long")], default="A scene, an idea, a place, anything is imaginable ... ")
    submit = SubmitField('visualize it')