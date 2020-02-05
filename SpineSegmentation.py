import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

vert_encode = {
  'T6': 1,
  'T7': 2,
  'T8': 3,
  'T9': 4,
  'T10': 5,
  'T11': 6,
  'T12': 7,

  'L1': 8,
  'L2': 9,
  'L3': 10,
  'L4': 11,
  'L5': 12}

vert_list = ['T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
             'L1', 'L2', 'L3', 'L4', 'L5']

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
    self.segVolumeNode = None
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

    # self.segVolumeTotalNode = slicer.vtkMRMLModelHierarchyNode()
    # self.segVolumeTotalNode.SetName('segVolumeTotalNode')
    # self.segVolumeTotalNode.SetSingletonTag('segVolumeTotalNode')
    # # self.segVolumeTotalNode.HideFromEditorsOn()
    # slicer.mrmlScene.AddNode(self.segVolumeTotalNode)

    self.segVolumeTotalNode = []
    self.segCombineVol = None

    inputCollapsibleButton = ctk.ctkCollapsibleButton()
    inputCollapsibleButton.text = "Input Volumes"
    self.layout.addWidget(inputCollapsibleButton)

    self.pathText = qt.QLineEdit()
    # self.__fixedShrinkFactor.setText("16, 16, 16")
    self.pathText.setToolTip('Specify output path')
    self.layout.addWidget(self.pathText)

    # Layout within the dummy collapsible button
    inputFormLayout = qt.QFormLayout(inputCollapsibleButton)


    loadSpineVolumeButton = qt.QPushButton('Load spine volume')
    loadSpineVolumeButton.connect('clicked()', self.loadVolume)
    inputFormLayout.addRow('Load spine volume', loadSpineVolumeButton)

    loadSegVolumeButton = qt.QPushButton('Load segmentation volume')
    loadSegVolumeButton.connect('clicked()', self.loadVolume)
    inputFormLayout.addRow('Load segmentation volume', loadSegVolumeButton)


    # input volume selector
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
    self.markerTableSelector.noneEnabled = True
    self.markerTableSelector.renameEnabled = False
    markerTable.setMRMLScene(slicer.mrmlScene)
    markerTable.setCurrentNode(slicer.mrmlScene.GetNodeByID(slicer.modules.markups.logic().AddNewFiducialNode()))
    markerTable.show()
    cropFormLayout.addWidget(markerTable)

    font = qt.QFont()
    font.setBold(True)

    # Segment vertebrae button

    self.segmentationButton = qt.QPushButton('Segment Vertebrae')
    self.segmentationButton.setFont(font)
    self.segmentationButton.toolTip = 'Segment the selected vertebrae'
    self.segmentationButton.enabled = False

    # Segmentation button connections
    self.segmentationButton.connect('clicked(bool)', self.onSegmentButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.markerTableSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    self.layout.addWidget(self.segmentationButton)





    # Combine Segmentations
    combineCollapsibleButton = ctk.ctkCollapsibleButton()
    combineCollapsibleButton.text = "Combine Segmentations"
    self.layout.addWidget(combineCollapsibleButton)

    # Layout within the dummy collapsible button
    combineFormLayout = qt.QFormLayout(combineCollapsibleButton)


    # Section to assign labels to segmentations and combine segmentations to a single volume
    self.L5SegSelector = DefaultSegSelect()
    self.L5SegSelector.setToolTip("Select L5 Segmentation")
    combineFormLayout.addRow("L5: ", self.L5SegSelector)
    # self.L5SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel(self.L5SegSelector.currentNode()))
    self.L5SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.L4SegSelector = DefaultSegSelect()
    self.L4SegSelector.setToolTip("Select L4 Segmentation")
    combineFormLayout.addRow("L4: ", self.L4SegSelector)
    self.L4SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.L3SegSelector = DefaultSegSelect()
    self.L3SegSelector.setToolTip("Select L3 Segmentation")
    combineFormLayout.addRow("L3: ", self.L3SegSelector)
    self.L3SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.L2SegSelector = DefaultSegSelect()
    self.L2SegSelector.setToolTip("Select L2 Segmentation")
    combineFormLayout.addRow("L2: ", self.L2SegSelector)
    self.L2SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.L1SegSelector = DefaultSegSelect()
    self.L1SegSelector.setToolTip("Select L1 Segmentation")
    combineFormLayout.addRow("L1: ", self.L1SegSelector)
    self.L1SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.T12SegSelector = DefaultSegSelect()
    self.T12SegSelector.setToolTip("Select T12 Segmentation")
    combineFormLayout.addRow("T12: ", self.T12SegSelector)
    self.T12SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.T11SegSelector = DefaultSegSelect()
    self.T11SegSelector.setToolTip("Select T11 Segmentation")
    combineFormLayout.addRow("T11: ", self.T11SegSelector)
    self.T11SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.T10SegSelector = DefaultSegSelect()
    self.T10SegSelector.setToolTip("Select T10 Segmentation")
    combineFormLayout.addRow("T10: ", self.T10SegSelector)
    self.T10SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.T9SegSelector = DefaultSegSelect()
    self.T9SegSelector.setToolTip("Select T9 Segmentation")
    combineFormLayout.addRow("T9: ", self.T9SegSelector)
    self.T9SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.T8SegSelector = DefaultSegSelect()
    self.T8SegSelector.setToolTip("Select T8 Segmentation")
    combineFormLayout.addRow("T8: ", self.T8SegSelector)
    self.T8SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.T7SegSelector = DefaultSegSelect()
    self.T7SegSelector.setToolTip("Select T7 Segmentation")
    combineFormLayout.addRow("T7: ", self.T7SegSelector)
    self.T7SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.T6SegSelector = DefaultSegSelect()
    self.T6SegSelector.setToolTip("Select T6 Segmentation")
    combineFormLayout.addRow("T6: ", self.T6SegSelector)
    self.T6SegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displayLabel)

    self.SegSelectList = [self.L5SegSelector,
                          self.L4SegSelector,
                          self.L3SegSelector,
                          self.L2SegSelector,
                          self.L1SegSelector,

                          self.T12SegSelector,
                          self.T11SegSelector,
                          self.T10SegSelector,
                          self.T9SegSelector,
                          self.T8SegSelector,
                          self.T7SegSelector,
                          self.T6SegSelector,
                          ]

    # Segmentation button connections
    self.combineSegButton = qt.QPushButton('Combine Segmentations')
    self.combineSegButton.toolTip = 'Combine Segmented Vertebrae'
    self.combineSegButton.connect('clicked(bool)', self.onCombineSegButton)
    self.combineSegButton.setFont(font)
    combineFormLayout.addRow(self.combineSegButton)





    # Reset segmentations and fiducial markers
    self.resetSegButton = qt.QPushButton('Reset Segmentation and Markers')
    self.resetSegButton.toolTip = "Reset fiducial markers and individual vertebrae segmentations"
    self.resetSegButton.setFont(font)
    self.resetSegButton.connect('clicked(bool)', self.onResetSegButton)
    self.layout.addWidget(self.resetSegButton)




    # Save segmentation button
    self.saveButton = qt.QPushButton('Save')
    self.saveButton.toolTip = "Save segmentation"
    self.saveButton.setFont(font)
    self.saveButton.connect('clicked(bool)', self.onSaveButton)
    combineFormLayout.addRow(self.saveButton)



    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def displayLabel(self, seg):
    # seg = self.L5SegSelector.currentNode()

    # seg =

    slicer.util.setSliceViewerLayers(label=seg, labelOpacity=1)


  def cleanup(self):
    pass

  def onSelect(self):
    self.segmentationButton.enabled = self.inputSelector.currentNode() and self.markerTableSelector.currentNode()

  def onResetSegButton(self):

    for segNode in self.segVolumeTotalNode:
      slicer.mrmlScene.RemoveNode(segNode)

    self.segVolumeTotalNode = []

    individualSeg = slicer.mrmlScene.GetNodesByName('segPrediction')
    num_of_seg = individualSeg.GetNumberOfItems()

    for idx in range(num_of_seg):
      slicer.mrmlScene.RemoveNode(individualSeg.GetItemAsObject(idx))


  def onSegmentButton(self):
    logic = SpineSegmentationLogic()
    self.segVolumeTotalNode = logic.run(self.inputSelector.currentNode(), self.markerTableSelector.currentNode(),
                                        self.segVolumeTotalNode)



  def onCombineSegButton(self):
    logic = CombineSegmentationLogic()
    self.segCombineVol = logic.run(self.SegSelectList, self.inputSelector.currentNode())


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


  def onSaveButton(self):
    # slicer.util.openSaveDataDialog()

    seg_filename = self.segCombineVol.GetName()

    save_dir = self.pathText.text

    if save_dir == '':
      print('No path has been entered')
      return

    if os.path.exists(save_dir) is False:
      print('Path entered does not exist. Please enter a proper path.')
      return

    slicer.util.saveNode(self.segCombineVol, save_dir + '/' + seg_filename + '.nii.gz')


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

  def run(self, inputVolume, fiducialMarker, segVolumeTotalNode):
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

    self.build_model()


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


      segVolumeNode = sitkUtils.PushVolumeToSlicer(y_pred_sitk_full_size,
                                                   name='segPrediction',
                                                   className='vtkMRMLLabelMapVolumeNode')

      # sitkUtils.PushVolumeToSlicer(y_pred_sitk_full_size, targetNode=segVolumeNode)

      segVolumeTotalNode.append(segVolumeNode)

      slicer.util.setSliceViewerLayers(label=segVolumeNode, labelOpacity=1)


    return segVolumeTotalNode

  def segment_vertebrae(self, spine_data):
    # spine_data[spine_data < -1024] = -1024
    spine_data = (spine_data - spine_data.min()) / (spine_data.max() - spine_data.min()) * 255

    x_data = np.expand_dims(np.expand_dims(spine_data, axis=0), axis=-1)

    y_pred = self.model.predict_on_batch(x_data)

    y_pred = np.where(y_pred > 0.5, 1, 0)[0, :, :, :, 0]

    return y_pred


  def build_model(self):
    hold = __file__
    hold2 = hold.split('/')
    model_files_root = ('/').join(hold2[:-1]) + '/SegUtils'
    # model_files_root = 'C:/SlicerExtension/SpineMets/SpineSegmentation/SegUtils'

    training_params_json = model_files_root + '/' + 'model_training_params.json'

    model_training_params_json = open(training_params_json, 'r')
    training_params = json.load(model_training_params_json)

    model_weights_filename = model_files_root + '/' + 'keras_unet_model_weights.h5'

    model_json_filename = model_files_root + '/' + 'keras_model.json'

    self.model = build_model(training_params,
                             model_weights_filename,
                             model_json_filename)

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


class CombineSegmentationLogic(ScriptedLoadableModuleLogic):
  def onClick(self):
    pass

  def run(self, SegSelectList, spineNode):
    seg_encode_list = []

    seg_full = None
    for idx, seg_select in enumerate(SegSelectList):

      seg_node = seg_select.currentNode()

      if seg_node is None:
        continue
      seg_img = sitkUtils.PullVolumeFromSlicer(seg_node)

      seg_img_encode = sitk.Multiply(seg_img, vert_encode[vert_list[idx]])

      if seg_full is None:
        seg_full = seg_img_encode

      else:
        seg_full = sitk.Add(seg_full, seg_img_encode)

    seg_name = spineNode.GetName()

    self.segCombineVol = sitkUtils.PushVolumeToSlicer(seg_full,
                                                      name=seg_name + '_seg',
                                                      className='vtkMRMLLabelMapVolumeNode')

    slicer.util.setSliceViewerLayers(label=self.segCombineVol, labelOpacity=1)


    return self.segCombineVol


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


def DefaultSegSelect():
  SegSelector = slicer.qMRMLNodeComboBox()
  SegSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
  SegSelector.selectNodeUponCreation = False
  SegSelector.addEnabled = False
  SegSelector.removeEnabled = False
  SegSelector.noneEnabled = True
  SegSelector.showHidden = False
  SegSelector.showChildNodeTypes = False
  SegSelector.setMRMLScene(slicer.mrmlScene)

  return SegSelector
