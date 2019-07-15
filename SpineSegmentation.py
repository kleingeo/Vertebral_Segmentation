import os
import unittest
import SimpleITK as sitk
import sitkUtils
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
from slicer.util import getNode

import json
import numpy as np
import keras
from keras import backend as K


from keras.models import model_from_json
from SegUtils.Utils import build_model

#
# SpineSegmentation
#

class SpineSegmentation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Spine Segmentation" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Spine Mets"]
    self.parent.dependencies = []
    self.parent.contributors = ["Geoff Klein (None)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.


    self.parent = parent
#
# SpineSegmentationWidget
#

class SpineSegmentationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):

    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    inputCollapsibleButton = ctk.ctkCollapsibleButton()
    inputCollapsibleButton.text = "Input Volumes"
    self.layout.addWidget(inputCollapsibleButton)

    # Layout within the dummy collapsible button
    inputFormLayout = qt.QFormLayout(inputCollapsibleButton)


    loadSpineVolumeButton = qt.QPushButton('Load spine volume')
    loadSpineVolumeButton.connect('clicked()', self.loadVolume)
    inputFormLayout.addRow('Load spine volume', loadSpineVolumeButton)

    loadSegVolumeButton = qt.QPushButton('Load segmentation volume')
    loadSegVolumeButton.connect('clicked()', self.loadVolume)
    inputFormLayout.addRow('Load segmentation volume', loadSegVolumeButton)

    # TODO Add functionality so that if the segmentation is NOT imported as a label map, it is converted to one immediately

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Select the CT spine volume." )
    inputFormLayout.addRow("Input Volume: ", self.inputSelector)

    #
    # output volume selector
    #
    self.segmentationSelector = slicer.qMRMLNodeComboBox()
    self.segmentationSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.segmentationSelector.selectNodeUponCreation = True
    self.segmentationSelector.addEnabled = True
    self.segmentationSelector.removeEnabled = True
    self.segmentationSelector.noneEnabled = True
    self.segmentationSelector.showHidden = False
    self.segmentationSelector.showChildNodeTypes = False
    self.segmentationSelector.setMRMLScene( slicer.mrmlScene )
    self.segmentationSelector.setToolTip( "Select the ground truth segmentation "
                                          "(if any exists) to compare with prediction." )
    inputFormLayout.addRow("Segmentation Volume: ", self.segmentationSelector)


    # TODO Add a save resampled volume


    cropCollapsibleButton = ctk.ctkCollapsibleButton()
    cropCollapsibleButton.text = "Select Vertebrae"
    self.layout.addWidget(cropCollapsibleButton)

    # Layout within the dummy collapsible button
    cropFormLayout = qt.QFormLayout(cropCollapsibleButton)



    markerTable = slicer.qSlicerSimpleMarkupsWidget()
    self.markerTableSelector = markerTable.MarkupsFiducialNodeComboBox
    self.markerTableSelector.selectNodeUponCreation = False
    self.markerTableSelector.addEnabled = True
    self.markerTableSelector.removeEnabled = True
    self.markerTableSelector.noneEnabled = False
    self.markerTableSelector.renameEnabled = True
    markerTable.setMRMLScene(slicer.mrmlScene)
    markerTable.setCurrentNode(slicer.mrmlScene.GetNodeByID(slicer.modules.markups.logic().AddNewFiducialNode()))
    markerTable.show()
    cropFormLayout.addWidget(markerTable)





    # # crop output volume selector
    # self.cropOutputSelector = slicer.qMRMLNodeComboBox()
    # self.cropOutputSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    # self.cropOutputSelector.toolTip = "Crop volumes for vertebrae levels"
    # self.cropOutputSelector.setMRMLScene(slicer.mrmlScene)
    # self.cropOutputSelector.renameEnabled = True
    # self.cropOutputSelector.addEnabled = True
    # self.cropOutputSelector.noneEnabled = True
    # self.cropOutputSelector.selectNodeUponCreation = True
    # self.cropOutputSelector.noneDisplay = 'Define name for cropped volume'
    # self.cropOutputSelector.removeEnabled = True
    # self.cropOutputSelector.showHidden = True
    #
    #
    # cropFormLayout.addRow("Output Crop Volume: ", self.cropOutputSelector)
    #
    #
    # # crop button
    # cropButton = qt.QPushButton("Crop Vertebrae")
    # cropButton.connect("clicked(bool)", self.onCropButton)
    # cropFormLayout.addRow("Crop Vertebrae", cropButton)


    # Segment vertebrae button
    self.segmentationButton = qt.QPushButton('Segment Vertebrae')
    self.segmentationButton.toolTip = 'Segment the selected vertebrae'
    self.segmentationButton.enabled = False

    # Segmentation button connections
    self.segmentationButton.connect('clicked(bool)', self.onSegmentButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.markerTableSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    self.layout.addWidget(self.segmentationButton)


    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.segmentationButton.enabled = self.inputSelector.currentNode() and self.markerTableSelector.currentNode()


  def onSegmentButton(self):
    logic = SpineSegmentationLogic()
    logic.run(self.inputSelector.currentNode(), self.markerTableSelector.currentNode())


  def loadVolume(self):
    slicer.util.openAddVolumeDialog()


  def onAddMarker(self):

    selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
    selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
    interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
    placeModePersistence = 1
    interactionNode.SetPlaceModePersistence(placeModePersistence)
    # mode 1 is Place, can also be accessed via slicer.vtkMRMLInteractionNode().Place
    interactionNode.SetCurrentInteractionMode(1)

# SpineSegmentationLogic
#


class SpineSegmentationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def onClick(self):
    pass

  def run(self, inputVolume, fiducialMarker):
    """
    Run the actual algorithm
    """

    inputVolume_origin = inputVolume.GetOrigin()
    inputVolume_spacing = inputVolume.GetSpacing()
    inputVolume_size = inputVolume.GetImageData().GetDimensions()

    spine_img = sitkUtils.PullVolumeFromSlicer(inputVolume)

    size_of_bbox = np.ceil(np.array([128, 128, 64]) / np.array(inputVolume_spacing)).astype(np.int)

    fiducial_coords_world_hold = [0, 0, 0, 0]
    numFids = fiducialMarker.GetNumberOfFiducials()

    for idx in range(numFids):

      fiducialMarker.GetNthFiducialWorldCoordinates(idx, fiducial_coords_world_hold)

      fiducial_coords_world = fiducial_coords_world_hold

      fiducial_coords_world[0] = fiducial_coords_world[0] * (-1)
      fiducial_coords_world[1] = fiducial_coords_world[1] * (-1)


      # Bounding box is [128, 128, 64] in mm with respect to the world coordinates

      fiducial_coords = np.floor(spine_img.TransformPhysicalPointToIndex(fiducial_coords_world[:3])).astype(np.int)

      ROI = sitk.RegionOfInterestImageFilter()
      ROI.SetSize([int(size_of_bbox[0]), int(size_of_bbox[1]), int(size_of_bbox[2])])
      ROI_initial_index = fiducial_coords - size_of_bbox/2
      ROI_initial_index = [roi_idx if roi_idx > 0 else 0 for roi_idx in ROI_initial_index]
      ROI.SetIndex([int(ROI_initial_index[0]), int(ROI_initial_index[1]), int(ROI_initial_index[2])])


      spine_img_cropped = ROI.Execute(spine_img)

      # Resample cropped spine image

      spacingOut = [1.0, 1.0, 1.0]
      resample = sitk.ResampleImageFilter()

      resample.SetReferenceImage(spine_img_cropped)
      resample.SetInterpolator(sitk.sitkLinear)

      shapeIn = spine_img_cropped.GetSize()
      spacingIn = spine_img_cropped.GetSpacing()

      newSize = [int(shapeIn[0] * spacingIn[0] / spacingOut[0]),
                 int(shapeIn[1] * spacingIn[1] / spacingOut[1]),
                 int(shapeIn[2] * spacingIn[2] / spacingOut[2])]

      resample.SetSize(newSize)

      resample.SetOutputSpacing(spacingOut)
      spine_img_resampled = resample.Execute(spine_img_cropped)

      # Second cropping to ensure image is the right size. Could be off my a 1 due to rounding.

      ROI = sitk.RegionOfInterestImageFilter()
      ROI.SetSize([128, 128, 64])
      ROI.SetIndex([0, 0, 0])

      spine_img_resampled = ROI.Execute(spine_img_resampled)

      # Get the spine data in a numpy array.
      spine_data = sitk.GetArrayFromImage(spine_img_resampled)

      y_pred_np = self.segment_vertebrae(spine_data)


      y_pred_sitk = sitk.GetImageFromArray(y_pred_np)
      y_pred_sitk.CopyInformation(spine_img_resampled)


      resample_back = sitk.ResampleImageFilter()

      resample_back.SetReferenceImage(spine_img)


      affine = sitk.AffineTransform(3)

      resample_back.SetTransform(affine)

      resample_back.SetInterpolator(sitk.sitkNearestNeighbor)
      y_pred_sitk_full_size = resample_back.Execute(y_pred_sitk)


      #self.seg_pred = sitkUtils.PushVolumeToSlicer(y_pred_sitk)

      self.segVolumeNode = sitkUtils.PushVolumeToSlicer(y_pred_sitk_full_size,
                                                        name='segPrediction',
                                                        className='vtkMRMLLabelMapVolumeNode')


    return True

  def segment_vertebrae(self, spine_data):
    hold = __file__
    hold2 = hold.split('/')
    model_files_root = ('/').join(hold2[:-1]) + '/SegUtils'
    # model_files_root = 'C:/SlicerExtension/SpineMets/SpineSegmentation/SegUtils'

    training_params_json = model_files_root + '/' + 'training_params_unet_True_True_dsc_True_1_True.json'

    model_training_params_json = open(training_params_json, 'r')
    training_params = json.load(model_training_params_json)

    model_weights_filename = model_files_root + '/' + 'unet_True_True_dsc_True_1_True_200_weights.h5'

    model_json_filename = model_files_root + '/' + 'model_json_unet_True_True_dsc_True_1_True.json'

    self.model = build_model(training_params,
                             model_weights_filename,
                             model_json_filename)

    # spine_data[spine_data < -1024] = -1024
    spine_data = (spine_data - spine_data.min()) / (spine_data.max() - spine_data.min()) * 255

    x_data = np.expand_dims(np.expand_dims(spine_data, axis=0), axis=-1)

    y_pred = self.model.predict_on_batch(x_data)

    y_pred = np.where(y_pred > 0.5, 1, 0)[0, :, :, :, 0]

    return y_pred


class SpineSegmentationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_SpineSegmentation1()

  def test_SpineSegmentation1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = SpineSegmentationLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')



def dice_coef(y_true, y_pred):
    """This method calculates the dice coefficient between the true
     and predicted masks
    Args:
        y_true: The true mask(i.e. ground-truth or expert annotated mask)
        y_pred: The predicted mask

    Returns:
        double: The dice coefficient"""

    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """This method calculates the loss function based on dice coeff
    Args:
        y_true: The true mask(i.e. ground truth or expert annotated mask)
        y_pred: The predicted mask

    Returns:
        double: The dice coefficient based loss function
    """
    return -dice_coef(y_true, y_pred)


def concurrency(y_true, y_pred):
    prediction_threshold = 0.5
    y_pred = K.cast(K.greater(y_pred, prediction_threshold), dtype="float32")

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return ((intersection / K.sum(y_true_f)) + (intersection / K.sum(y_pred_f))) / 2.