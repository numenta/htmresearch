// ----------------------------------------------------------------------
//  Copyright (C) 2006-2008 Numenta Inc. All rights reserved.
//
//  The information and source code contained herein is the
//  exclusive property of Numenta Inc. No part of this software
//  may be used, reproduced, stored or distributed in any form,
//  without explicit written authorization from Numenta Inc.
// ----------------------------------------------------------------------

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <sys/syslimits.h>


#define MAXLABEL    9

typedef struct _LABEL_HEADER {
  int    nMagic;
  int    nNumLabels;
} LABEL_HEADER;

typedef struct _IMAGE_HEADER {
  int    nMagic;
  int    nNumImages;
  int    nNumRows;
  int    nNumCols;
} IMAGE_HEADER;


int _readAndFlipByte(FILE * fp) {
  int nByte;
  int nNumRead = fread(&nByte, sizeof(nByte), 1, fp);
  assert(nNumRead == 1);
  return ntohl(nByte);
}


void _createDirIfNeeded(const char * szDirPath) {
  // Create output directory
  struct stat sStatBuf;
  int nPerms = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
  int nError = stat(szDirPath, &sStatBuf);
  if (nError  ||  !(sStatBuf.st_mode & S_IFDIR)) {
    nError = mkdir(szDirPath, nPerms);
    if (nError) {
      fprintf(stdout, "Could not create directory: %s [0x%08x]\n", szDirPath, nError);
      exit(-1);
    }
  }
}


void _doProcessing(const char * szImagePath,
                   const char * szLabelPath,
                   const char * szDisposition) {

  int k;
  int nNumRead;

  //------------------------------------------------
  // Label file format:
  //
  //  [offset] [type]          [value]          [description]
  //  0000     32 bit integer  0x00000801(2049) magic number (MSB first)
  //  0004     32 bit integer  60000            number of items
  //  0008     unsigned byte   ??               label
  //  0009     unsigned byte   ??               label
  //  ........
  //  xxxx     unsigned byte   ??               label
  //
  //  The labels values are 0 to 9. 
  //

  // Open label files
  FILE * fpLabel = fopen(szLabelPath, "rb");
  assert(fpLabel);

  // Read header
  LABEL_HEADER sLabelHeader;
  sLabelHeader.nMagic = _readAndFlipByte(fpLabel);
  assert(sLabelHeader.nMagic == 0x00000801);
  sLabelHeader.nNumLabels = _readAndFlipByte(fpLabel);

  // Allcoate memory
  fprintf(stdout, "Labels found: %d\n", sLabelHeader.nNumLabels);
  unsigned char * pcLabel = new unsigned char[sLabelHeader.nNumLabels];
  assert(pcLabel);

  // Read all (big-endian) bytes
  nNumRead = fread(pcLabel, sizeof(*pcLabel), sLabelHeader.nNumLabels, fpLabel);
  assert(nNumRead == sLabelHeader.nNumLabels);

  unsigned int anCounts[MAXLABEL+1];
  memset(anCounts, 0x0, sizeof(anCounts));

  // Check each byte
  unsigned char *pcLabelPtr = pcLabel;
  for (k=sLabelHeader.nNumLabels; k; k--) {
    assert(*pcLabelPtr <= 9);
    anCounts[*pcLabelPtr]++;
    pcLabelPtr++;
  }

  for (k=0; k<=MAXLABEL; k++)
    fprintf(stdout, "[%d]: %d\n", k, anCounts[k]);

  // Close files
  fclose(fpLabel);


  //------------------------------------------------
  // Image file format:
  //
  //  [offset] [type]          [value]          [description]
  //  0000     32 bit integer  0x00000803(2051) magic number
  //  0004     32 bit integer  60000            number of images
  //  0008     32 bit integer  28               number of rows
  //  0012     32 bit integer  28               number of columns
  //  0016     unsigned byte   ??               pixel
  //  0017     unsigned byte   ??               pixel
  //  ........
  //  xxxx     unsigned byte   ??               pixel
  //

  // Open image files
  FILE * fpImage = fopen(szImagePath, "rb");
  assert(fpImage);

  // Read header
  IMAGE_HEADER sImageHeader;
  sImageHeader.nMagic = _readAndFlipByte(fpImage);
  assert(sImageHeader.nMagic == 0x00000803);
  sImageHeader.nNumImages = _readAndFlipByte(fpImage);
  sImageHeader.nNumRows = _readAndFlipByte(fpImage);
  sImageHeader.nNumCols = _readAndFlipByte(fpImage);

  // Allcoate memory
  fprintf(stdout, "Images found: %d\n", sImageHeader.nNumImages);
  int nNumPixels = sImageHeader.nNumImages * sImageHeader.nNumRows * sImageHeader.nNumCols;
  unsigned char * pcImage = new unsigned char[nNumPixels];
  assert(pcImage);

  // Read all bytes
  nNumRead = fread(pcImage, sizeof(*pcImage), nNumPixels, fpImage);
  assert(nNumRead == nNumPixels);
  fprintf(stdout, "Pixels read: %d\n", nNumRead);

  // Close files
  fclose(fpImage);


  //------------------------------------------------------
  // Generate output images

  // Create output directory
  _createDirIfNeeded(szDisposition);

  // Create each sub-output directory
  char szDstPath[PATH_MAX+1];
  for (k=0; k<=MAXLABEL; k++) {
    sprintf(szDstPath, "%s/%d", szDisposition, k);
    _createDirIfNeeded(szDstPath);
  }

  // Write each image
  int i, j;
  int nLabel;
  int nError = 0;
  //int nBytesPerImage = sImageHeader.nNumRows * sImageHeader.nNumCols;
  const unsigned char * pcImagePtr = (const unsigned char *)pcImage;
  memset(anCounts, 0x0, sizeof(anCounts));
  for (k=0; k<sImageHeader.nNumImages; k++) {
    nLabel = pcLabel[k]; 
    sprintf(szDstPath, "%s/%d/%06d.txt", szDisposition, nLabel, anCounts[nLabel]);
    fprintf(stdout, "Writing: %s\n", szDstPath);
    anCounts[nLabel]++;
    FILE * fpOut = fopen(szDstPath, "w");
    assert(fpOut);
    fprintf(fpOut, "%d %d\n", sImageHeader.nNumRows, sImageHeader.nNumCols);
    for (j=sImageHeader.nNumRows; j; j--) {
      for (i=sImageHeader.nNumCols; i; i--)
        fprintf(fpOut, "%d ", *pcImagePtr++);
      fprintf(fpOut, "\n");
    }
    fclose(fpOut);
    fpOut = NULL;
  }

  //------------------------------------------------------
  // Free memory
  if (pcLabel) {
    delete [] pcLabel;
    pcLabel = NULL;
  }
}


int main(int argc, char * argv[]) {
  
  const char * szTrainImagePath = "train-images-idx3-ubyte";
  const char * szTrainLabelPath = "train-labels-idx1-ubyte";
  const char * szTestImagePath = "t10k-images-idx3-ubyte";
  const char * szTestLabelPath = "t10k-labels-idx1-ubyte";
  
  // Process training/testing data
  _doProcessing(szTrainImagePath, szTrainLabelPath, "training");
  _doProcessing(szTestImagePath, szTestLabelPath, "testing");
}
