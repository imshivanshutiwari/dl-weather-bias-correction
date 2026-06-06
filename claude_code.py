const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat,
  TabStopType, TabStopPosition, ImageRun, HeightRule
} = require('docx');
const fs = require('fs');

// ─── PALETTE ────────────────────────────────────────────────────────────────
const NAVY      = "0D1F3C";   // deep navy
const GOLD      = "C9A84C";   // gold accent
const SLATE     = "3A5F7D";   // mid blue
const MIST      = "EEF3F8";   // light bg
const MIST2     = "F7F9FB";   // alternating row bg
const CHARCOAL  = "1E1E1E";   // body text
const WHITE     = "FFFFFF";
const DIVIDER   = "C9A84C";   // gold divider
const BORDER    = "D0DBE5";
const FONT      = "Times New Roman";

// ─── MARGINS (unchanged from original) ──────────────────────────────────────
// top=0.3125", bottom=0", left=0.5", right=0.5"  => DXA
const TOP_M    = 100; // was 450
const BOT_M    = 0;
const LR_M     = Math.round(0.5 * 1440);    // 720
// Content width = 8.5" - 0.5" - 0.5" = 7.5" = 10800 DXA
const CONTENT_W = 10800;

// ─── BULLET CONFIG ────────────────────────────────────────────────────────
const BULLET_REF = "main-bullets";

// ─── HELPERS ────────────────────────────────────────────────────────────────

function goldDivider() {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 12, color: GOLD, space: 1 }
    },
    children: []
  });
}

function thinDivider() {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 4, color: BORDER, space: 1 }
    },
    children: []
  });
}

function spacer(before = 120, after = 120) {
  return new Paragraph({ spacing: { before, after }, children: [] });
}

function h1(text) {
  return [
    new Paragraph({
      spacing: { before: 360, after: 60 },
      children: [
        new TextRun({
          text: "◆  " + text,
          font: FONT,
          size: 38,        // 19pt
          bold: true,
          color: NAVY,
        })
      ]
    }),
    new Paragraph({
      spacing: { before: 0, after: 180 },
      border: {
        bottom: { style: BorderStyle.SINGLE, size: 20, color: GOLD, space: 4 }
      },
      children: []
    })
  ];
}

function h2(text) {
  return new Paragraph({
    spacing: { before: 260, after: 100 },
    children: [
      new TextRun({
        text: "▸  " + text,
        font: FONT,
        size: 28,   // 14pt
        bold: true,
        color: SLATE,
      })
    ]
  });
}

function body(text, { bold = false, italic = false } = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 60, after: 120, line: 276 },
    children: [
      new TextRun({ text, font: FONT, size: 22, bold, italic, color: CHARCOAL })
    ]
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: BULLET_REF, level: 0 },
    spacing: { before: 40, after: 80, line: 264 },
    children: [
      new TextRun({ text, font: FONT, size: 22, color: CHARCOAL })
    ]
  });
}

function callout(title, text) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: BORDER };
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: CONTENT_W, type: WidthType.DXA },
            shading: { fill: "FDF8EE", type: ShadingType.CLEAR },
            borders: {
              top:    { style: BorderStyle.SINGLE, size: 1,  color: GOLD },
              bottom: { style: BorderStyle.SINGLE, size: 1,  color: GOLD },
              left:   { style: BorderStyle.SINGLE, size: 36, color: GOLD },
              right:  { style: BorderStyle.NONE,   size: 0,  color: WHITE }
            },
            margins: { top: 200, bottom: 200, left: 280, right: 200 },
            children: [
              new Paragraph({
                spacing: { before: 0, after: 80 },
                children: [
                  new TextRun({ text: "★  " + title + ":  ", font: FONT, size: 21, bold: true, color: NAVY }),
                  new TextRun({ text, font: FONT, size: 21, italic: true, color: "555555" })
                ]
              })
            ]
          })
        ]
      })
    ]
  });
}

// ─── HEADER / FOOTER ────────────────────────────────────────────────────────
function makeHeader() {
  let iitm_hdr, diat_hdr;
  try {
    const iitm_data = fs.readFileSync("E:\\iitm pune project important files\\MY FILES\\iitm pune.png");
    const diat_data = fs.readFileSync("E:\\iitm pune project important files\\MY FILES\\Defence_Institute_of_Advanced_Technology.png");
    iitm_hdr = new ImageRun({ data: iitm_data, transformation: { width: 55, height: 55 }, type: "png" });
    diat_hdr = new ImageRun({ data: diat_data, transformation: { width: 55, height: 55 }, type: "png" });
  } catch (e) {
    iitm_hdr = new TextRun({ text: "" });
    diat_hdr = new TextRun({ text: "" });
  }

  return new Header({
    children: [
      new Table({
        width: { size: CONTENT_W, type: WidthType.DXA },
        columnWidths: [CONTENT_W - 2000, 2000],
        borders: {
          top: { style: BorderStyle.NIL, size: 0, color: "auto" },
          bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 4 },
          left: { style: BorderStyle.NIL, size: 0, color: "auto" },
          right: { style: BorderStyle.NIL, size: 0, color: "auto" },
          insideHorizontal: { style: BorderStyle.NIL, size: 0, color: "auto" },
          insideVertical: { style: BorderStyle.NIL, size: 0, color: "auto" }
        },
        rows: [
          new TableRow({
            children: [
              new TableCell({
                verticalAlign: VerticalAlign.BOTTOM,
                margins: { left: 0, right: 0, top: 0, bottom: 0 },
                children: [
                  new Paragraph({
                    spacing: { before: 0, after: 0 },
                    children: [
                      new TextRun({ text: "DEEP LEARNING APPROACHES FOR WEATHER DATA ENHANCEMENT", font: FONT, size: 16, color: SLATE, bold: true })
                    ]
                  })
                ]
              }),
              new TableCell({
                verticalAlign: VerticalAlign.BOTTOM,
                margins: { left: 0, right: 0, top: 0, bottom: 0 },
                children: [
                  new Paragraph({
                    alignment: AlignmentType.RIGHT,
                    spacing: { before: 0, after: 0 },
                    children: [
                      iitm_hdr,
                      new TextRun({ text: " " }),
                      diat_hdr
                    ]
                  })
                ]
              })
            ]
          })
        ]
      })
    ]
  });
}

function makeFooter() {
  return new Footer({
    children: [
      new Paragraph({
        spacing: { before: 60, after: 0 },
        border: {
          top: { style: BorderStyle.SINGLE, size: 4, color: GOLD, space: 4 }
        },
        tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
        children: [
          new TextRun({ text: "Confidential Technical Document", font: FONT, size: 16, color: SLATE }),
          new TextRun({ text: "\t", font: FONT, size: 16 }),
          new TextRun({ text: "Page ", font: FONT, size: 16, color: CHARCOAL }),
          new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, bold: true, color: NAVY }),
        ]
      })
    ]
  });
}

// ─── TABLE HELPER ────────────────────────────────────────────────────────────
function buildTable(headers, rows, colWidths) {
  const hdrBorder = { style: BorderStyle.SINGLE, size: 1, color: NAVY };
  const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: BORDER };

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) =>
      new TableCell({
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: NAVY, type: ShadingType.CLEAR },
        borders: { top: hdrBorder, bottom: hdrBorder, left: hdrBorder, right: hdrBorder },
        margins: { top: 120, bottom: 120, left: 150, right: 150 },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: h, font: FONT, size: 20, bold: true, color: WHITE })]
        })]
      })
    )
  });

  const dataRows = rows.map((row, ri) =>
    new TableRow({
      children: row.map((cell, ci) =>
        new TableCell({
          width: { size: colWidths[ci], type: WidthType.DXA },
          shading: { fill: ri % 2 === 0 ? WHITE : MIST2, type: ShadingType.CLEAR },
          borders: {
            top: cellBorder, bottom: cellBorder,
            left: ci === 0 ? { style: BorderStyle.SINGLE, size: 16, color: GOLD } : cellBorder,
            right: cellBorder
          },
          margins: { top: 100, bottom: 100, left: 130, right: 130 },
          children: [new Paragraph({
            children: [new TextRun({
              text: cell,
              font: FONT,
              size: 18,
              bold: ci === 0,
              color: ci === 0 ? NAVY : CHARCOAL
            })]
          })]
        })
      )
    })
  );

  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
}

// ─── TITLE PAGE ──────────────────────────────────────────────────────────────
function titlePage() {
  const borderDouble = { style: BorderStyle.DOUBLE, size: 18, color: NAVY };
  const borderGoldSingle = { style: BorderStyle.SINGLE, size: 8, color: GOLD };
  
  let iitm, diat;
  try {
    iitm = fs.readFileSync("E:\\iitm pune project important files\\MY FILES\\iitm pune.png");
    diat = fs.readFileSync("E:\\iitm pune project important files\\MY FILES\\Defence_Institute_of_Advanced_Technology.png");
  } catch (e) {
    // ignore
  }

  const innerW = 10240; // width of inner border box
  const contentW = innerW - 800; // 9440 DXA

  // Institutional Header Table
  const headerTable = new Table({
    width: { size: contentW, type: WidthType.DXA },
    columnWidths: [1100, contentW - 2200, 1100],
    borders: {
      top: { style: BorderStyle.NIL, size: 0, color: "auto" },
      bottom: { style: BorderStyle.NIL, size: 0, color: "auto" },
      left: { style: BorderStyle.NIL, size: 0, color: "auto" },
      right: { style: BorderStyle.NIL, size: 0, color: "auto" },
      insideHorizontal: { style: BorderStyle.NIL, size: 0, color: "auto" },
      insideVertical: { style: BorderStyle.NIL, size: 0, color: "auto" }
    },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            verticalAlign: VerticalAlign.CENTER,
            margins: { left: 0, right: 0, top: 0, bottom: 0 },
            children: [
              iitm ? new Paragraph({
                alignment: AlignmentType.LEFT,
                children: [new ImageRun({ data: iitm, transformation: { width: 90, height: 90 }, type: "png" })]
              }) : new Paragraph({ children: [] })
            ]
          }),
          new TableCell({
            verticalAlign: VerticalAlign.CENTER,
            margins: { left: 100, right: 100, top: 0, bottom: 0 },
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  new TextRun({ text: "INDIAN INSTITUTE OF TROPICAL METEOROLOGY, PUNE", font: FONT, size: 22, bold: true, color: NAVY })
                ]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 40 },
                children: [
                  new TextRun({ text: "&", font: FONT, size: 22, bold: true, color: SLATE })
                ]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 40 },
                children: [
                  new TextRun({ text: "DEFENCE INSTITUTE OF ADVANCED TECHNOLOGY, PUNE", font: FONT, size: 20, bold: true, color: SLATE })
                ]
              })
            ]
          }),
          new TableCell({
            verticalAlign: VerticalAlign.CENTER,
            margins: { left: 0, right: 0, top: 0, bottom: 0 },
            children: [
              diat ? new Paragraph({
                alignment: AlignmentType.RIGHT,
                children: [new ImageRun({ data: diat, transformation: { width: 90, height: 90 }, type: "png" })]
              }) : new Paragraph({ children: [] })
            ]
          })
        ]
      })
    ]
  });

  // Title Banner
  const titleBanner = new Table({
    width: { size: contentW, type: WidthType.DXA },
    columnWidths: [contentW],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            shading: { fill: NAVY, type: ShadingType.CLEAR },
            borders: {
              top: borderGoldSingle,
              bottom: borderGoldSingle,
              left: borderGoldSingle,
              right: borderGoldSingle
            },
            margins: { top: 400, bottom: 400, left: 300, right: 300 },
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 0, after: 120 },
                children: [
                  new TextRun({ text: "◆  ◆  ◆", font: FONT, size: 20, color: GOLD })
                ]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 100, after: 120 },
                children: [new TextRun({
                  text: "DEEP LEARNING APPROACHES FOR",
                  font: FONT, size: 40, bold: true, color: WHITE
                })]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 0, after: 160 },
                children: [new TextRun({
                  text: "WEATHER DATA ENHANCEMENT",
                  font: FONT, size: 40, bold: true, color: GOLD
                })]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 0, after: 0 },
                border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: GOLD, space: 1 } },
                children: []
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 180, after: 140 },
                children: [new TextRun({
                  text: "A Comprehensive Spatiotemporal Framework for Bias Correction (CNNBC) and High-Fidelity Downscaling (SRGAN) over the Indian Monsoon Region",
                  font: FONT, size: 20, italic: true, color: "AABCCE"
                })]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 120, after: 0 },
                children: [new TextRun({
                  text: "A Technical Book, Master Thesis & Secure Defense Operational Integration Manual",
                  font: FONT, size: 18, bold: true, color: WHITE
                })]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 0, after: 40 },
                children: [
                  new TextRun({ text: "◆  ◆  ◆", font: FONT, size: 20, color: GOLD })
                ]
              })
            ]
          })
        ]
      })
    ]
  });

  // Inner Box containing the content, with double navy border
  const innerBox = new Table({
    width: { size: innerW, type: WidthType.DXA },
    columnWidths: [innerW],
    borders: {
      top: borderDouble,
      bottom: borderDouble,
      left: borderDouble,
      right: borderDouble,
      insideHorizontal: { style: BorderStyle.NIL, size: 0, color: "auto" },
      insideVertical: { style: BorderStyle.NIL, size: 0, color: "auto" }
    },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: innerW, type: WidthType.DXA },
            margins: { top: 500, bottom: 500, left: 400, right: 400 },
            children: [
              headerTable,
              spacer(400),
              titleBanner,
              spacer(500),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: "PREPARED BY", font: FONT, size: 18, color: SLATE, italic: true })]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 120, after: 120 },
                children: [new TextRun({ text: "SHIVANSHU TIWARI", font: FONT, size: 32, bold: true, color: NAVY })]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: "◆  ◆  ◆", font: FONT, size: 16, color: GOLD })]
              }),
              spacer(300),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: "Developed in Collaboration with IITM Pune & Academic Partners", font: FONT, size: 18, color: "555555" })]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 100 },
                children: [new TextRun({ text: "2025 - 2026", font: FONT, size: 16, color: "777777", bold: true })]
              })
            ]
          })
        ]
      })
    ]
  });

  // Outer Box spanning the entire physical page (12240 x 15840)
  return [
    new Table({
      width: { size: 12240, type: WidthType.DXA },
      columnWidths: [12240],
      borders: {
        top: { style: BorderStyle.NIL, size: 0, color: "auto" },
        bottom: { style: BorderStyle.NIL, size: 0, color: "auto" },
        left: { style: BorderStyle.NIL, size: 0, color: "auto" },
        right: { style: BorderStyle.NIL, size: 0, color: "auto" },
        insideHorizontal: { style: BorderStyle.NIL, size: 0, color: "auto" },
        insideVertical: { style: BorderStyle.NIL, size: 0, color: "auto" }
      },
      rows: [
        new TableRow({
          height: { value: 15840, rule: HeightRule.EXACT },
          children: [
            new TableCell({
              width: { size: 12240, type: WidthType.DXA },
              shading: { fill: MIST, type: ShadingType.CLEAR },
              margins: { top: 350, bottom: 800, left: 1000, right: 1000 },
              verticalAlign: VerticalAlign.TOP,
              children: [
                innerBox
              ]
            })
          ]
        })
      ]
    })
  ];
}

// ─── CHAPTER LABEL BLOCK ────────────────────────────────────────────────────
function chapterLabel(num, title) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: BORDER };
  return [
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [800, CONTENT_W - 800],
      rows: [
        new TableRow({
          children: [
            new TableCell({
              width: { size: 800, type: WidthType.DXA },
              shading: { fill: GOLD, type: ShadingType.CLEAR },
              borders: { top: border, bottom: border, left: border, right: { style: BorderStyle.NONE, size: 0, color: WHITE } },
              margins: { top: 160, bottom: 160, left: 160, right: 160 },
              verticalAlign: VerticalAlign.CENTER,
              children: [new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [new TextRun({ text: String(num).padStart(2, "0"), font: FONT, size: 48, bold: true, color: NAVY })]
              })]
            }),
            new TableCell({
              width: { size: CONTENT_W - 800, type: WidthType.DXA },
              shading: { fill: NAVY, type: ShadingType.CLEAR },
              borders: { top: border, bottom: border, right: border, left: { style: BorderStyle.NONE, size: 0, color: WHITE } },
              margins: { top: 160, bottom: 160, left: 240, right: 160 },
              verticalAlign: VerticalAlign.CENTER,
              children: [
                new Paragraph({ children: [new TextRun({ text: "CHAPTER", font: FONT, size: 18, color: GOLD, bold: true })] }),
                new Paragraph({ spacing: { before: 60 }, children: [new TextRun({ text: title, font: FONT, size: 30, bold: true, color: WHITE })] })
              ]
            })
          ]
        })
      ]
    }),
    spacer(180, 60)
  ];
}

// ─── ALL CONTENT ─────────────────────────────────────────────────────────────
const children = [];

// push helper
function push(...items) {
  for (const i of items) Array.isArray(i) ? push(...i) : children.push(i);
}

// TITLE (Handled in Section 1 of the Document configuration)

// ABSTRACT
push(...h1("Abstract"));
push(body("Weather forecasting plays an indispensable role in maintaining the security, economic stability, and agricultural output of a nation. The Global Forecast System (GFS) is a robust numerical weather prediction system utilized worldwide. However, when applied to geophysically complex and highly volatile regions such as the Indian subcontinent, the GFS model output inherently suffers from two severe limitations: systematic meteorological biases and coarse spatial resolution. These limitations severely hamper the precision required for critical, hyper-local applications ranging from disaster management to tactical military operations."));
push(body("This thesis introduces an advanced, dual-phase deep learning architecture specifically engineered to sequentially solve these two fundamental problems. The first phase focuses on the absolute mitigation of systematic errors using a specialized 3D Convolutional Neural Network for Bias Correction (CNNBC), incorporating residual averaging and grouped convolutions to handle 4D spatio-temporal meteorological tensors. The second phase dramatically enhances the spatial fidelity of the unbiased data using a Super-Resolution Generative Adversarial Network (SRGAN). The generator network leverages deep residual blocks and PixelShuffle upsampling to achieve extreme upscaling factors of up to 25x, converting coarse 0.625-degree data into hyper-local 0.025-degree intelligence. The models are rigorously validated against the Indian Monsoon Data Assimilation and Analysis (IMDAA) dataset using PSNR, MSE, MAE, and Pearson Correlation Coefficients."));
push(body("Furthermore, this comprehensive document details the extensive dataset engineering pipeline, mathematical formulations, software ecosystem prerequisites, and crucially, an exhaustive analysis of how this exact framework can be seamlessly transitioned into a mission-critical defense and tactical military forecasting system. Finally, an extensive Handover Guide provides explicit documentation of every script, folder, and algorithm to ensure seamless continuity for future researchers."));
push(new Paragraph({ children: [new PageBreak()] }));

// TABLE OF CONTENTS
push(...h1("Table of Contents"));
const tocItems = [
  ["Chapter 1", "Introduction and Background", "Background on NWP models, GFS limitations, and the Deep Learning paradigm shift."],
  ["Chapter 2", "Technical Background and Theoretical Foundations", "Deep dive into 3D CNNs, Generative Adversarial Networks, Residual learning, and PixelShuffle."],
  ["Chapter 3", "Study Area and Dataset Engineering", "Defining the geographic domain, GFS parsing, land-sea masking, and IMDAA ground-truth preparation."],
  ["Chapter 4", "Comprehensive File and Folder Handover Guide", "Extensive documentation of every script, library dependency, input, and output folder in the repository."],
  ["Chapter 5", "The Bias Correction Framework (CNNBC)", "Mathematical modeling, the ResCNNv4 architecture, grouped convolutions, and PReLU activation."],
  ["Chapter 6", "Spatial Downscaling Framework (SRGAN)", "The Generator and Discriminator architecture, minimax game, and PixelShuffle upsampling blocks."],
  ["Chapter 7", "Baseline Models and Evaluation Metrics", "VGG baseline comparison and quantitative metrics: PSNR, MSE, MAE, and Pearson Correlation."],
  ["Chapter 8", "Experimental Results and Visualizations", "Statistical probability densities (KDE), Cartopy spatial plots, and monsoon validation figures."],
  ["Chapter 9", "Prerequisite Tools, Libraries, and Study Guide", "Full study guide for PyTorch, Xarray, PyGrib, Cartopy, NumPy, and Pandas for future maintainers."],
  ["Chapter 10", "Defense and Military Applications (Special Section)", "Tactical forecasting, UAV/Drone strike coordination, artillery fire correction, and edge deployment."],
  ["Chapter 11", "Extensive Future Scope and Scaling", "Multi-modal variables, ONNX/TensorRT optimization, Vision Transformers, and Continuous MLOps."],
  ["Chapter 12", "Conclusion", "Summary of project achievements and structural handover overview."],
];
push(buildTable(["#", "Chapter Title", "Focus Area"], tocItems, [720, 4320, 5760]));
push(spacer());
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 1 ───────────────────────────────────────────────────────────────
push(...chapterLabel(1, "Introduction and Background"));
push(h2("1.1 The Importance of Meteorological Accuracy"));
push(body("The modern world operates on the back of accurate meteorological data. Ranging from civilian aviation routing to agricultural planning, and extending deep into military logistics, the necessity for precise weather prediction cannot be overstated. Traditional forecasting relies on Numerical Weather Prediction (NWP) models, which simulate atmospheric fluid dynamics using vast arrays of supercomputers. The Global Forecast System (GFS), operated by the United States National Weather Service, is one of the most prominent NWP models globally. It utilizes complex differential equations governing thermodynamics, fluid dynamics, and atmospheric chemistry to predict the future state of the atmosphere."));
push(body("However, physics-based simulations require vast computational resources. To run globally, the GFS divides the earth into discrete 3D grid boxes. The size of these grid boxes dictates the resolution. If a grid box is 70 kilometers wide, the model calculates one single average value for temperature, pressure, and rainfall for that entire 70-kilometer zone, completely ignoring smaller geographic features like individual valleys, hills, or urban heat islands."));
push(h2("1.2 Inherent Limitations of the GFS Model"));
push(body("Despite its massive computational backing, the GFS is a global model, meaning its primary objective is to simulate weather across the entire planet. This macro-scale focus inherently sacrifices micro-scale precision. When zoomed into geophysically diverse areas like the Indian subcontinent—a region dominated by the massive Himalayan range, extensive coastlines, and the highly chaotic Indian Monsoon system—the GFS output breaks down in accuracy."));
push(body("This breakdown manifests primarily as Systematic Bias. The physical equations governing the GFS cannot perfectly capture sub-grid level phenomena, leading to continuous overestimations or underestimations of rainfall. For instance, if a mountain is smaller than the grid box, the model will not accurately simulate the orographic lifting of air, causing it to consistently under-predict rainfall on the windward side of that mountain. Because this error is systematic (it happens the exact same way every time given the same conditions), it is a prime target for machine learning."));
push(body("A second, equally critical limitation is Spatial Resolution. GFS outputs are provided at coarse grids (e.g., 0.25 to 0.625 degrees). At a 0.625-degree resolution, a single data pixel covers an area roughly 70x70 kilometers. It is impossible to make tactical or highly localized decisions when entire cities are encompassed by a single data point. Predicting precise flooding in a specific district of Mumbai is impossible using raw GFS data."));
push(h2("1.3 The Deep Learning Paradigm Shift"));
push(body("Recent advancements in artificial intelligence, specifically in the domains of Computer Vision and Deep Learning, have opened revolutionary new pathways to augment numerical weather predictions. By treating meteorological fields as multi-dimensional images, researchers can apply Convolutional Neural Networks (CNNs) to find complex, non-linear relationships that physical models miss."));
push(body("This project proposes a dual-phase deep learning pipeline. Instead of attempting to replace the GFS, this pipeline acts as an intelligent post-processing augmentation layer. It takes the coarse, biased GFS data as input, mathematically strips away the historical bias using a 3D Residual CNN, and subsequently hallucinates hyper-realistic, mathematically sound sub-grid details using a Generative Adversarial Network (GAN). The final output is an ultra-high-resolution, zero-bias weather map."));
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 2 ───────────────────────────────────────────────────────────────
push(...chapterLabel(2, "Technical Background and Theoretical Foundations"));
push(h2("2.1 Convolutional Neural Networks (CNNs) in Meteorology"));
push(body("Convolutional Neural Networks have dominated the field of image processing by utilizing localized convolutional filters to extract spatial hierarchies of features. In meteorological contexts, a 2D CNN can capture spatial pressure gradients, temperature fronts, and storm cell structures. However, weather is not static; it is a highly fluid spatio-temporal phenomenon."));
push(body("To capture the evolution of weather, this project heavily utilizes 3D Convolutional Neural Networks. A Conv3D layer extends the sliding window mechanism across a third dimension—time. By feeding the network consecutive forecast steps or multiple forecast ensembles, the Conv3D filters learn the temporal velocity and acceleration of weather systems, providing a far more robust bias correction mechanism than standard 2D approaches."));
push(body("Mathematically, a 3D convolution operates by sliding a filter of size (Time, Height, Width) across the input tensor. At each step, it calculates the dot product between the filter weights and the local input data. This allows the network to recognize patterns not just in a static image, but across the fluid movement of the storm systems over hours and days."));
push(h2("2.2 Generative Adversarial Networks (GANs)"));
push(body("The challenge of spatial downscaling (super-resolution) is inherently an ill-posed mathematical problem; there are infinitely many high-resolution images that could theoretically down-sample to the exact same low-resolution input. Standard CNNs trained on Mean Squared Error (MSE) tend to output the mathematical average of all possible solutions, resulting in highly blurred, smoothed-out weather maps that completely lack intense, localized weather events (like sudden cloudbursts)."));
push(body("To counter this, the project utilizes a Generative Adversarial Network (GAN). Proposed by Ian Goodfellow in 2014, a GAN consists of two networks locked in a zero-sum game:"));
push(bullet("The Generator (G): Attempts to synthesize ultra-high-resolution weather maps from coarse inputs. It is parameterized by weights theta_G."));
push(bullet("The Discriminator (D): Attempts to differentiate between true high-resolution observational data and the synthetic data produced by the Generator. It is parameterized by weights theta_D."));
push(body("The adversarial loss forces the Generator to move beyond calculating mathematical averages. It must generate data that possesses the exact same high-frequency textural statistics as real weather data, resulting in incredibly sharp and realistic meteorological downscaling. The minimax game ensures that at equilibrium, the generated distribution is indistinguishable from the true data distribution."));
push(h2("2.3 Residual Learning and PixelShuffle"));
push(body("Deep networks are notoriously difficult to train due to the vanishing gradient problem. As gradients are backpropagated through dozens of layers, they diminish exponentially, making early layers impossible to train. This project utilizes Residual Networks (ResNets), which employ skip-connections that bypass non-linear layers. This allows gradients to flow unimpeded deep into the network. Mathematically, instead of learning a direct mapping H(x), the network learns a residual mapping F(x) = H(x) - x, and then outputs F(x) + x."));
push(body("For the upsampling phase, rather than using traditional Transpose Convolutions (which cause severe checkerboard artifacts and mathematical distortion), the architecture employs Sub-Pixel Convolution (PixelShuffle). PixelShuffle rearranges elements from the depth channel (feature maps) directly into spatial dimensions. Specifically, it reshapes a tensor of shape (B, C * r^2, H, W) into (B, C, H*r, W*r), where r is the upscaling factor. This ensures mathematically flawless and artifact-free upscaling."));
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 3 ───────────────────────────────────────────────────────────────
push(...chapterLabel(3, "Study Area and Dataset Engineering"));
push(h2("3.1 Geographic and Temporal Domain"));
push(body("The study is rigorously constrained to the Indian Subcontinent. The bounding box coordinates are: Latitude 6.5° N to 38.51° N, and Longitude 66.5° E to 100.1° E, evaluated at 0.125-degree increments."));
push(body("Temporally, the data spans from 2019 to 2023, specifically isolating the Indian Monsoon season (JJAS: June, July, August, September). The monsoon period represents the highest volatility and complexity in atmospheric science, making it the ultimate testbed for deep learning stability. Capturing the variance of extreme rainfall events during these months guarantees the model is robust enough for year-round deployment."));
push(h2("3.2 Predictor Dataset: GFS Processing Pipeline"));
push(body("The Global Forecast System (GFS) supplies the raw predictor inputs. The pipeline handles this through the Preprocess_GFS_data.py and analyse_grib.py scripts."));
push(bullet("Extraction: Raw GRIB files are parsed using the pygrib library. Variables are isolated into NumPy arrays. GRIB is a highly compressed binary format used by meteorological organizations. Decoding it requires precision to ensure no coordinate shifting occurs."));
push(bullet("Masking: A highly specific geographical mask (Mask_125deg_New.npy) is broadcast across the entire dataset. This mask isolates the exact domain of interest and converts all irrelevant geographical grid points (such as deep ocean areas unrelated to the mainland forecast) to NaN (Not a Number). This acts as a massive computational optimizer, allowing the neural networks to strictly ignore dead space and focus their weights entirely on the complex topography of India."));
push(bullet("Temporal Accumulation: GFS data often represents instantaneous or short-interval rates. The scripts programmatically sum these intervals to generate 3-hourly accumulated rainfall tensors, aligning with the ground truth temporal resolution. This accumulation logic must account for leap years and missing data packets."));
push(bullet("Tensor Formatting: The final output is structured into a massive 4-Dimensional NetCDF tensor comprising (Time, Ensemble, Latitude, Longitude). Including the 10 forecast ensembles allows the bias correction model to evaluate forecast uncertainty and consensus, which is a powerful predictor of absolute accuracy."));
push(h2("3.3 Ground Truth Dataset: IMDAA Processing"));
push(body("The Indian Monsoon Data Assimilation and Analysis (IMDAA) dataset serves as the absolute target (label) for the neural networks. Processed via Preprocess_IMDAA.py, the APCP_sfc (Accumulated Precipitation at Surface) variable is extracted."));
push(body("To achieve perfect spatial synchronization with the GFS domain, the IMDAA data must be interpolated. The script utilizes the Python Imaging Library (PIL) to treat the numerical arrays as images, resizing them to a strict 269x257 grid. Crucially, Nearest Neighbor interpolation is enforced. Unlike bilinear interpolation, nearest neighbor prevents the mathematical fabrication of decimal rainfall values, preserving the exact observational physics recorded by the IMDAA systems. If a specific coordinate recorded 0mm of rain, bilinear interpolation might accidentally smooth it to 0.5mm based on surrounding pixels. Nearest Neighbor prevents this data corruption."));
push(h2("3.4 Climatology Integration"));
push(body("The script calculate_climatology.py scans multi-year historical data to catalog maximum possible rainfall values per coordinate. This json-based climatology matrix provides the neural networks with a physical upper-bound ceiling, preventing the Generative Adversarial Network from hallucinating physically impossible extreme rainfall values. By normalizing data against climatology, the model learns relative extremeness rather than absolute arbitrary bounds."));
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 4 ───────────────────────────────────────────────────────────────
push(...chapterLabel(4, "Comprehensive File and Folder Handover Guide"));
push(body("To ensure that any researcher, data scientist, or defense engineer can seamlessly assume control of this project, this chapter provides a granular, file-by-file breakdown of the entire workspace hierarchy. Every script's purpose, input, and output is documented."));
push(spacer(100, 140));

const handoverRows = [
  ["/Preprocess_datasets/Preprocess_GFS_data.py", "Parses NCEP GRIB, applies Mask_125deg_New.npy, aggregates to 3h.", "Raw GFS GRIB folder", "GFS_daily_2019to23.nc"],
  ["/Preprocess_datasets/Preprocess_IMDAA.py", "Extracts APCP_sfc, resizes via PIL.Image.NEAREST to 269x257.", "IMDAA GRIB/NC files", "IMDAA_3h.nc"],
  ["/Preprocess_datasets/calculate_climatology.py", "Computes spatiotemporal historical max ceiling values.", "Multi-year GFS NC", "GFS_3h_max_climatology.json"],
  ["/Preprocess_datasets/Make_Timestamps.py", "Synchronizes datetime64 indexes for dataloaders.", "Datetime configs", "time_stamps.npy"],
  ["/CNNBC/model_cnnbc.py", "Earlier 3D CNN model using spatiotemporal rainfall bias averaging.", "PyTorch model template", "cnnbc_weights.pth"],
  ["/CNNBC/cnnbc_final.py", "End-to-end training pipeline, evaluates raw vs corrected arrays.", "GFS_daily_2019to23.nc", "Inference loss logs"],
  ["/Bias_Correction_2025/model_4.py", "Finalized ResCNNv4 class with 8 residual blocks and grouped convs.", "Grouped 10-channel tensors", "ResCNNv4 class module"],
  ["/Bias_Correction_2025/train_cnn.py", "Executes multi-GPU training with DataParallel and AMP autocast.", "GFS_daily_2019to23.nc", "best_model_cnnbc.pth"],
  ["/Bias_Correction_2025/config.py", "Holds training configurations: BATCH_SIZE=128, LR=1e-8.", "Path strings", "Configuration variables"],
  ["/Bias_Correction_2025/cnnbc_utils.py", "Calculates custom PSNR and handles EarlyStopper thresholds.", "Loss history", "Early stop trigger signals"],
  ["/Downscaling/models_final.py", "Defines the Generator and Discriminator neural architectures.", "GELU/LeakyReLU maps", "GAN network objects"],
  ["/Downscaling/train_gen.py", "Performs generator pre-training utilizing MSE losses.", "Coarse training tensors", "pretrained_generator.pth"],
  ["/Downscaling/train_disc.py", "Executes the minimax adversarial game utilizing BCE and GAN losses.", "IMDAA target arrays", "final_srgan_weights.pth"],
  ["/Downscaling/vgg/VGG.py", "Defines baseline 1-channel modified VGG19 architecture.", "1-channel weather grids", "VGG19 network object"],
  ["/Downscaling/vgg/train.py", "Trains and logs baseline VGG19 metrics for comparative study.", "Rainfall arrays", "vgg_weights.pth"],
  ["/Figures/Report_figures/", "Stores empirical correlation and spatial performance maps.", "Inference statistics", "KDE, Cartopy, scatter PNGs"],
];
push(buildTable(["Script / File Path", "Functional Purpose", "Inputs Ingested", "Outputs Emitted"], handoverRows, [2400, 3200, 2600, 2600]));
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 5 ───────────────────────────────────────────────────────────────
push(...chapterLabel(5, "The Bias Correction Framework (CNNBC)"));
push(h2("5.1 The Concept of Meteorological Bias"));
push(body("Systematic bias in numerical weather models is a deterministic error. If a mountain range is not properly resolved by the coarse 70km grid of the GFS, the model will consistently fail to simulate orographic lifting, leading to a permanent underestimation of rainfall on the windward side of the mountain. Because this error is systematic and bound to geographical topology, a deep neural network can easily learn to reverse it."));
push(h2("5.2 The ResCNNv4 Architecture"));
push(body("The core intelligence of the bias correction phase is encapsulated in model_4.py, which defines the ResCNNv4 class."));
push(body("The architecture takes in a tensor with in_channels=10 (representing the 10 GFS ensembles)."));
push(bullet("Initial Feature Extraction: The data passes through a massive 9x9 spatial convolution. This large kernel size allows the network to observe broad, synoptic-scale weather systems spanning hundreds of kilometers simultaneously. The activation function used is PReLU (Parameterized ReLU), which allows the network to learn a dynamic alpha parameter for negative values, ensuring gradients do not die during extended dry periods (zero rainfall)."));
push(bullet("Grouped Convolutions: The layers utilize groups=10. This forces the network to apply convolutions independently across the different ensembles before mixing them. It acts as an implicit regularizer, preventing extreme overfitting on small dataset variances while severely reducing the parameter count and VRAM footprint."));
push(bullet("Residual Core: The network processes features through a deep core of 8 Residual_Block modules. Each block contains convolutions, and crucially, an additive skip connection return x + self.conv_block(x). This allows deep feature extraction without losing the original baseline topology of the weather map."));
push(h2("5.3 Training Methodology and Optimization"));
push(body("The training script (train_cnn.py) is engineered for extreme high-performance computing. It utilizes torch.nn.DataParallel to distribute the massive tensor calculations across multiple GPUs."));
push(body("The optimization is governed by the Adam optimizer with a highly conservative learning rate of 1e-8, designed for extreme micro-adjustments during late-stage training. To prevent Out-Of-Memory (OOM) errors and drastically accelerate throughput, the script utilizes torch.cuda.amp.autocast(), enabling Automatic Mixed Precision (AMP). This allows the GPUs to perform calculations in 16-bit precision where possible, doubling the processing speed and halving the VRAM requirements without sacrificing mathematical stability."));
push(body("The model optimizes against Mean Squared Error (MSE), aggressively punishing outlier errors to ensure the model respects extreme, catastrophic weather anomalies."));
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 6 ───────────────────────────────────────────────────────────────
push(...chapterLabel(6, "Spatial Downscaling Framework (SRGAN)"));
push(h2("6.1 Super-Resolution in Meteorology"));
push(body("After the CNNBC removes all systematic biases, the data is highly accurate but still trapped at a coarse 0.625-degree resolution. The Spatial Downscaling phase utilizes a Super-Resolution Generative Adversarial Network (SRGAN) to upscale this data by an astonishing factor of up to 25x, reaching a hyper-local 0.025-degree resolution."));
push(h2("6.2 The Generator Network (models_final.py)"));
push(body("The Generator is responsible for hallucinating the sub-grid meteorological details. The architecture is a massive feed-forward convolutional engine:"));
push(bullet("Feature Extraction: A 9x9 convolution begins the process, extracting the foundational topology of the coarse weather map. Large kernels are required here to map long-range dependencies across the coarse input."));
push(bullet("Deep Residual Trunk: The core of the generator consists of 8 massive Residual Blocks. Each block contains two 3x3 convolutions, separated by Batch Normalization layers to stabilize the internal covariate shift, and activated by GELU (Gaussian Error Linear Units), which provide smoother gradients than standard ReLU."));
push(bullet("PixelShuffle Upsampling: To achieve the 25x upscale factor, the generator utilizes 5 consecutive nn.PixelShuffle layers. Each PixelShuffle layer expands the spatial dimension by 2x. By stacking 5 of these layers, the network achieves a theoretical maximum upscale factor of 32x (2^5). The network comfortably maps the target 25x resolution within this space by cropping and padding mathematically."));
push(h2("6.3 The Discriminator Network"));
push(body("The Discriminator ensures the Generator does not create physically impossible weather artifacts. It is structured as a deep, strided convolutional classifier."));
push(body("Instead of using Max-Pooling, which destroys spatial relationships and topological structure, the Discriminator uses convolutions with stride=2 to halve the image dimensions while doubling the filter count (from 64 up to 512). It utilizes LeakyReLU activations (with a negative slope of 0.2) to maintain gradient flow even for negative inputs. The final layer is an AdaptiveAvgPool2d followed by dense linear layers that map the vast feature space down to a single binary probability scalar: 1 for Real (True IMDAA High-Resolution data) and 0 for Fake (Generator synthesized data)."));
push(h2("6.4 Adversarial Training Loop"));
push(body("The SRGAN utilizes a highly complex dual-training loop (train_disc.py and train_gen.py). The Generator is first pre-trained purely on Mean Squared Error (MSE) to learn the basic mapping. Once stabilized, the Adversarial loop begins. The Discriminator is trained using Binary Cross-Entropy (BCE) to detect fakes, and the Generator is trained using a composite loss function: a weighted sum of MSE Loss (for general structure) and Adversarial Loss (to trick the discriminator into outputting a 1). This mathematical tug-of-war forces the Generator to output stunningly sharp, physically realistic high-frequency weather features."));
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 7 ───────────────────────────────────────────────────────────────
push(...chapterLabel(7, "Baseline Models and Evaluation Metrics"));
push(h2("7.1 VGG Baseline Classification"));
push(body("To scientifically validate the supremacy of the ResCNNv4 and SRGAN models, a baseline architecture was developed using the classic VGG19 network (vgg/VGG.py). While historically used for ImageNet classifications, this architecture was heavily customized to ingest 1-channel meteorological tensors. Implementing a baseline proves that the complexity of residual connections, grouped convolutions, and adversarial learning are absolutely necessary for fluid dynamic data, as standard deep stacks (VGG) fail to match the performance metrics and suffer from vanishing gradients when dealing with the extreme sparsity of rainfall data."));
push(h2("7.2 Mathematical Evaluation Metrics"));
push(body("The framework is judged against a ruthless suite of statistical metrics:"));
push(bullet("Peak Signal-to-Noise Ratio (PSNR): The ultimate metric for the SRGAN downscaling. Calculated as 10 * log10(MAX^2 / MSE). Higher PSNR values indicate the synthesized high-resolution data is almost pixel-perfect compared to the IMDAA ground truth. Values exceeding 40 dB are considered exceptional."));
push(bullet("Mean Squared Error (MSE): By squaring the errors, MSE applies an exponential penalty to large mistakes. This is vital in meteorology, as failing to predict a massive cloudburst is significantly worse than slightly missing a light drizzle."));
push(bullet("Mean Absolute Error (MAE): Provides a linear understanding of error, useful for calculating total volume discrepancies in accumulated rainfall over an entire season."));
push(bullet("Pearson Correlation Coefficient: Measures the linear correlation between the generated dataset and the ground truth. A value close to 1.0 proves the statistical distributions and standard deviations of the AI model perfectly match the real world."));
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 8 ───────────────────────────────────────────────────────────────
push(...chapterLabel(8, "Experimental Results and Visualizations"));
push(h2("8.1 Statistical Validations and Probability Densities"));
push(body("The massive amount of empirical data generated during the evaluation phase is stored within the MY FILES/Figures/Report_figures/ directory. The evaluation metrics unequivocally prove the success of the architecture."));
push(bullet("Correlation KDE Plots: The Kernel Density Estimation plots (correlationKDE_set0.png, etc.) visually map the probability density function of the model output versus the ground truth. The overlapping peaks prove that the CNNBC effectively eliminated the systematic biases of the GFS, pulling the skewed GFS distribution directly in line with the IMDAA observational reality."));
push(bullet("Scatter Plot Validations: The correlation plots (correlationPlot_set0.png, etc.) show the tight clustering of AI predictions along the 45-degree y=x identity line, proving that the model did not just learn to predict average rainfall, but accurately scaled with intense rainfall events."));
push(h2("8.2 Geographic Spatial Consistency"));
push(body("Scripts utilizing the cartopy library generated high-resolution geographical maps (e.g., India2023_07_04.png). These visual checks act as the final confirmation that the SRGAN correctly understood topographical boundaries, correctly rendering sharp gradients across the Himalayan foothills and the Western Ghats without mathematical blurring or bleeding artifacts into the oceanic mask regions."));
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 9 ───────────────────────────────────────────────────────────────
push(...chapterLabel(9, "Prerequisite Study Guide and Learning Roadmap"));
push(body("To fully comprehend, deploy, extend, or debug this deep learning meteorological framework, a new researcher or defense engineer must deeply study and master a specific stack of scientific concepts, data engineering tools, and PyTorch paradigms. This chapter provides a highly granular, step-by-step learning roadmap. You must study these topics in this exact order to prevent massive skill gaps."));

push(h2("9.1 Foundational Theories to Master First"));
push(body("Before reading a single line of code in this repository, the following theoretical concepts must be crystal clear:"));
push(bullet("1. The Mathematics of Convolutional Neural Networks (CNNs): Understand 2D and 3D Convolutions, kernels, padding, stride, and how sliding windows extract feature maps. Understand why 3D convolutions are needed for Spatio-Temporal data (like weather) over 2D."));
push(bullet("2. Residual Learning (ResNets): Study how skip-connections (adding the input of a block to its output) solve the Vanishing Gradient problem in extremely deep networks. Read the original 2015 ResNet paper by Kaiming He."));
push(bullet("3. Generative Adversarial Networks (GANs): Study the minimax game theory. Understand how a Generator creates fake data and a Discriminator tries to catch it. Understand the difference between Binary Cross Entropy (BCE) Loss for the Discriminator and combined Adversarial/MSE Loss for the Generator."));
push(bullet("4. Sub-Pixel Convolution (PixelShuffle): Do not study 'Transpose Convolutions' as they cause checkerboard artifacts. Instead, study PixelShuffle, which reorganizes depth channels into spatial resolution for mathematically flawless upscaling."));

push(h2("9.2 Core Deep Learning Framework: PyTorch"));
push(body("This entire project is constructed upon the PyTorch ecosystem. To master this codebase, one must deeply study the following PyTorch mechanics:"));
push(bullet("Tensor Memory Management & CUDA: Understand exactly how to move vast multi-dimensional tensors between System RAM (CPU) and GPU VRAM using .to(device). Mismanagement will lead to immediate OOM (Out Of Memory) failures."));
push(bullet("Custom nn.Module Classes: You must study Object-Oriented deep learning. Understand how the __init__ function defines the layer weights (like nn.Conv2d, nn.BatchNorm2d) and how the forward(self, x) function defines the computational graph dynamically."));
push(bullet("DataLoaders & Datasets: Study the torch.utils.data.Dataset class. You need to know how to override the __len__ and __getitem__ functions to load massive NetCDF files chunk-by-chunk from the hard drive, rather than loading them entirely into RAM."));
push(bullet("High-Performance Training (AMP): One must study PyTorch Automatic Mixed Precision (torch.cuda.amp.autocast and GradScaler). This allows the GPU to calculate in 16-bit precision, doubling the speed and halving VRAM usage."));
push(bullet("Distributed Training: Study torch.nn.DataParallel to understand how the training scripts distribute massive batches across multiple GPUs simultaneously."));

push(h2("9.3 Geospatial and Meteorological Data Formats"));
push(body("Meteorological data is vastly more complex than standard CSV files or JPEG images. You must understand the file formats before manipulating them:"));
push(bullet("GRIB2 Format: Gridded Binary (GRIB) is the World Meteorological Organization's standard for storing weather data. It is highly compressed and binary-encoded. You must understand how GRIB messages store separate variables (like pressure, temperature) at specific pressure levels."));
push(bullet("NetCDF Format (Network Common Data Form): The standard format for storing multi-dimensional scientific data (Time, Latitude, Longitude, Ensembles). It stores metadata directly inside the file."));

push(h2("9.4 Essential Python Libraries to Master"));
push(body("You must master the following libraries to handle the data engineering and visualization pipelines:"));
push(bullet("Xarray (import xarray as xr): The most critical data engineering library. It manipulates labeled multi-dimensional arrays (NetCDF files). Study how to slice datasets along named dimensions (data.sel(lat=..., lon=...)) and how to compute geographical means."));
push(bullet("PyGrib (import pygrib): Essential for reading the raw GFS GRIB files. Study how to iterate through grib messages, read the keys, and extract specific variable arrays securely into NumPy matrices."));
push(bullet("Cartopy: Required for visualizing the data correctly mapped to earth projections. You must understand PlateCarree and Mercator projections, and how to overlay coastlines(), borders(), and shapefiles onto matplotlib figures to draw the Indian map accurately."));
push(bullet("NumPy (import numpy as np): Deep understanding of array broadcasting, boolean masking (np.where, np.isnan), and axis-manipulations (transposing, rolling) is strictly required for the preprocessing scripts."));
push(bullet("Pandas (import pandas as pd): Used heavily in Make_Timestamps.py for generating complex pd.date_range arrays to synchronize exact hours across multi-year spans. Study Pandas datetime functionality deeply."));
push(bullet("Pillow (from PIL import Image): Required for understanding the specific Image.NEAREST interpolation technique used during the IMDAA dataset spatial alignment to prevent fabricating non-existent rainfall data."));
push(bullet("Weights & Biases (import wandb): Used for real-time tracking of training losses, learning rates, and gradient norms across the cloud. Study how to initialize a wandb run and log metrics dictionaries."));

push(h2("9.5 Project-Specific Logic to Review"));
push(body("Once the foundational theories and libraries are mastered, the researcher should study the project scripts in this exact order:"));
push(bullet("1. Data Engineering: Read Preprocess_GFS_data.py to see how raw GRIBs become NetCDFs."));
push(bullet("2. Baseline: Read vgg/VGG.py to understand a simple classification network before looking at the complex ones."));
push(bullet("3. Bias Correction: Read Bias_Correction_2025/model_4.py to see how the ResCNNv4 uses grouped convolutions (groups=10) to process 10 ensembles independently."));
push(bullet("4. Super Resolution: Read Downscaling/models_final.py. Focus first on the Generator's Residual Blocks, then the PixelShuffle layers, and finally the Discriminator's strided convolutions."));
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 10 ──────────────────────────────────────────────────────────────
push(...chapterLabel(10, "Defense and Military Applications (Special Section)"));
push(body("The architecture described in this thesis—capable of completely neutralizing numerical forecasting bias and achieving hyper-local spatial resolution up to 25x—represents a massive tactical advantage. This section exhaustively details how this scientific academic codebase can be directly transitioned into a classified, mission-critical defense forecasting system."));
push(spacer(80, 120));
push(callout("TACTICAL VALUE PROPOSITION", "Standard global GFS models providing 70x70 kilometer resolution are tactically useless; an entire mountain valley, containing multiple infantry divisions, forward operating bases, and artillery batteries, easily fits within a single 70km pixel. A commanding officer cannot make life-or-death deployment decisions when the forecast averages the weather of a 20,000-foot mountain peak and the valley floor together. This section outlines the classified integration blueprint."));
push(spacer(120, 80));
push(h2("10.1 The Tactical Necessity of Hyper-Local Weather Intelligence"));
push(body("Military operations, particularly in complex topographies like the Himalayas, the Line of Actual Control (LAC), or coastal naval bases, are entirely at the mercy of the weather. Standard global GFS models providing 70x70 kilometer resolution are tactically useless; an entire mountain valley, containing multiple infantry divisions, forward operating bases, and artillery batteries, easily fits within a single 70km pixel. A commanding officer cannot make life-or-death deployment decisions when the forecast averages the weather of a 20,000-foot mountain peak and the valley floor together into one meaningless number."));
push(body("By applying the SRGAN downscaling to 0.025 degrees (roughly 2.5x2.5 kilometers), commanders gain hyper-local visibility. This allows for valley-specific, ridge-specific, and base-specific weather forecasting, an unprecedented advantage in modern asymmetrical warfare."));
push(h2("10.2 Application: Aviation and Drone Strike Operations"));
push(body("Unmanned Aerial Vehicles (UAVs) and fighter squadrons require absolute precision regarding cloud cover, precipitation, and thermal conditions. Sudden, unpredicted localized squalls or cloudbursts can ground operations, blind optical targeting pods, or cause catastrophic mission failure."));
push(body("Integration Strategy: The CNNBC+SRGAN framework can be deployed directly into Air Force command systems. By feeding the biased GFS data through the local edge-servers running the PyTorch models, flight planners receive unbiased, high-resolution precipitation fields. This allows them to thread flight paths through micro-scale clear-weather windows that standard models would simply show as a massive, impenetrable storm front."));
push(h2("10.3 Application: Artillery and Ballistic Trajectory Correction"));
push(body("Atmospheric density and heavy precipitation drastically alter the trajectory and drag coefficients of long-range artillery shells and ballistic missiles. Current targeting computers rely on localized weather balloons or coarse regional forecasts, leading to Circular Error Probabilities (CEP) that are larger than desired."));
push(body("Integration Strategy: By integrating this AI-augmented weather data directly into the fire-control computer systems of artillery batteries, targeting algorithms can account for hyper-localized precipitation walls between the firing position and the target. The bias correction phase ensures that systematic pressure/temperature errors are stripped out, guaranteeing higher first-strike accuracy."));
push(h2("10.4 Application: Naval Operations and Coastal Defense"));
push(body("Coastal boundaries are historically the hardest zones to forecast due to the complex interaction between sea breezes, ocean temperatures, and land topography."));
push(body("Integration Strategy: The SRGAN can be specifically re-trained using localized naval radar data as the ground truth, replacing IMDAA. This would allow the Navy to take standard, coarse global oceanic forecasts and downscale them into ultra-precise coastal wave and squall predictions. This is vital for amphibious landings, submarine surfacing operations, carrier group deployments, and protecting coastal installations from sudden tempest strikes."));
push(h2("10.5 Transitioning to a Secure Defense Project"));
push(body("To convert this academic codebase into a deployable defense asset, the following strict engineering steps must be taken:"));
push(bullet("Edge Deployment Optimization: The PyTorch models (.pth files) must be compiled using NVIDIA TensorRT or ONNX Runtime. Python is too slow for tactical deployment. Compiling to TensorRT allows the AI models to execute in C++ on ruggedized, low-power tactical servers deployed directly in the field, without requiring an active internet connection to central cloud servers."));
push(bullet("Air-Gapped Data Pipelines: The Preprocess_GFS_data.py scripts must be refactored to ingest raw GRIB files transmitted via secure military satellite uplinks (e.g., GSAT), completely isolating the system from public internet APIs and protecting it from cyber-warfare data-poisoning attacks."));
push(bullet("Expansion to Multivariate Tensors: The current project focuses exclusively on precipitation (APCP_sfc). For a complete defense suite, the input tensors must be expanded to include Wind Velocity (U and V vectors), Barometric Pressure, and Cloud Ceiling Heights. The in_channels of the ResCNNv4 architecture would simply be expanded to accommodate these variables simultaneously."));
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 11 ──────────────────────────────────────────────────────────────
push(...chapterLabel(11, "Extensive Future Scope and Scaling"));
push(body("The framework developed in this thesis represents a fundamental breakthrough, yet it serves merely as the foundational bedrock for what is possible. The future scope of this project is vast, requiring multi-disciplinary scaling across data engineering, deep learning architecture, and operational MLOps deployment. This chapter exhaustively outlines the specific roadmaps required to scale this project into a globally dominant forecasting system."));
push(h2("11.1 Multi-Modal Ensemble Expansion"));
push(body("The current iteration of the CNNBC and SRGAN models operates on a singular meteorological variable: accumulated precipitation. However, the atmosphere is a chaotic, fully coupled fluid dynamic system. Precipitation does not occur in a vacuum; it is the direct result of complex interactions between temperature gradients, barometric pressure drops, atmospheric moisture content, and wind shear vectors."));
push(body("Future scaling requires expanding the input tensors from a shape of (Ensembles, Spatial_X, Spatial_Y) to (Variables, Ensembles, Spatial_X, Spatial_Y). The models must be re-engineered to ingest GFS data for Temperature at 2m, Geopotential Heights at 500hPa, U-Wind and V-Wind components at various pressure levels, and Relative Humidity. By feeding the 3D CNNs a multi-modal view of the atmosphere, the network will no longer be guessing rainfall based purely on coarse rainfall estimates; it will be learning the actual physics of storm formation. The network will see the pressure drop and wind convergence, allowing it to predict a rainfall event even if the raw GFS rainfall forecast entirely missed it. This requires massive upgrades to the preprocessing scripts (re-writing Xarray pipelines to handle terabytes of multi-variable NetCDF data) and significant expansions to the ResCNNv4 input layers."));
push(h2("11.2 Global Real-Time Edge Deployment (MLOps)"));
push(body("Currently, the system is a highly successful academic proof-of-concept operating on historical, static datasets (2019-2023). To realize its full potential, it must be transitioned into a live, real-time Machine Learning Operations (MLOps) pipeline."));
push(body("The future scope demands the creation of an automated ingestion engine. This engine would utilize automated cron jobs or Apache Airflow DAGs to continuously poll the NOAA/NCEP servers. The moment a new GFS forecast GRIB file is published, the pipeline would automatically download it, push it through the Preprocess_GFS_data.py logic, execute the ResCNNv4 bias correction inference, pass the tensor to the SRGAN for 25x downscaling, and instantly publish the hyper-local forecast to a secure web dashboard or mobile API."));
push(body("Furthermore, to run inferencing at scale, the models cannot remain in pure PyTorch Python scripts. The future scope requires compiling the models into ONNX (Open Neural Network Exchange) format or utilizing NVIDIA TensorRT. This allows the models to run in highly optimized C++ or Rust environments, dropping inference time from seconds down to milliseconds. This optimization is what would allow the system to be deployed on Edge Devices—such as remote weather stations, naval ships, or agricultural IoT hubs—where computational power is strictly limited."));
push(h2("11.3 Next-Generation Architectures: Transformers and Diffusion"));
push(body("While the ResCNNv4 and SRGAN models represent state-of-the-art for current convolutional approaches, the field of Deep Learning moves exponentially. The future scope of this project involves replacing or augmenting the convolutional backbones with Next-Generation architectures."));
push(bullet("Vision Transformers (ViTs): CNNs are limited by their kernel size (e.g., 9x9). A CNN can only see a small patch of the weather map at any given time. If a storm system over the Bay of Bengal is heavily influenced by a high-pressure system over the Arabian Sea, a CNN might struggle to make that distant connection. Vision Transformers utilize Self-Attention mechanisms, allowing every single pixel on the map to mathematically attend to every other pixel, regardless of distance. Upgrading the Bias Correction module to a Swin-Transformer (Shifted Window Transformer) would allow the model to understand global synoptic patterns instantly, drastically improving accuracy."));
push(bullet("Denoising Diffusion Probabilistic Models (DDPMs): GANs are notoriously difficult to train, prone to mode collapse where the generator outputs the exact same image repeatedly. Diffusion models, which power modern image generators like Midjourney and DALL-E, offer a much more mathematically stable approach to generation. The future scope involves replacing the SRGAN with a Conditional Diffusion Model. The model would start with pure Gaussian noise and slowly denoise it, conditioned on the coarse GFS input, to produce the ultra-high-resolution output. Diffusion models have shown unprecedented success in generating highly complex, realistic textures, which is exactly what is needed for hyper-local terrain-based rainfall maps."));
push(h2("11.4 Automated Retraining and Concept Drift Mitigation"));
push(body("The global climate is rapidly changing due to anthropogenic global warming. The statistical distribution of the Indian Monsoon in 2030 will not be the same as it was in 2020. This phenomenon is known in machine learning as Concept Drift. If the models are trained on 2019-2023 data and never updated, their accuracy will slowly degrade as the actual climate shifts away from their training baseline."));
push(body("The ultimate future scope is the implementation of an Automated Continuous Learning pipeline. As new IMDAA ground truth data becomes available each month, the system would automatically compare its past predictions to the new ground truth. If the error (MSE) crosses a defined threshold, the system would automatically spin up cloud GPU instances, append the new data to the training set, re-train the models utilizing transfer learning (updating only the final layers to adapt to the new climate reality), and hot-swap the newly trained weights into the production environment with zero downtime. This ensures the forecasting system remains infinitely adaptable and perfectly accurate, regardless of how the global climate shifts in the coming decades."));
push(new Paragraph({ children: [new PageBreak()] }));

// ─── CHAPTER 12 ──────────────────────────────────────────────────────────────
push(...chapterLabel(12, "Conclusion"));
push(body("This thesis has successfully demonstrated that the fusion of 3D Residual Convolutional Neural Networks and Super-Resolution Generative Adversarial Networks provides a definitive, highly robust solution to the systematic biases and coarse resolutions inherent in global numerical weather prediction models."));
push(body("Operating within the extremely volatile and mathematically complex domain of the Indian Monsoon, the framework proved its absolute supremacy. The CNNBC module successfully neutralized persistent, geographically locked errors by learning the complex non-linear relationships between the biased forecast and the true ground reality. Subsequently, the SRGAN module mathematically deduced and rendered hyper-realistic sub-grid details up to an unprecedented 25x enhancement factor, proving that deep learning can hallucinate physically accurate weather data purely from coarse predictors."));
push(body("By exhaustively documenting the file structures, data engineering pipelines, mathematical architectures, and vast military and tactical applications, this book serves not only as a record of academic achievement but as a definitive blueprint for the future of localized atmospheric intelligence. The methodologies proven here pave the way for a new era where precision forecasting is no longer limited by the computational bottlenecks of supercomputers, but rather bounded only by the depth of the neural networks deployed."));

// ─── ASSEMBLE DOCUMENT ───────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: BULLET_REF,
      levels: [{
        level: 0,
        format: LevelFormat.BULLET,
        text: "◆",
        alignment: AlignmentType.LEFT,
        style: {
          paragraph: { indent: { left: 540, hanging: 360 } },
          run: { font: FONT, color: GOLD, size: 18 }
        }
      }]
    }]
  },
  styles: {
    default: {
      document: { run: { font: FONT, size: 22, color: CHARCOAL } }
    }
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 0, bottom: 0, left: 0, right: 0 }
        }
      },
      headers: {
        default: new Header({ children: [] })
      },
      footers: {
        default: new Footer({ children: [] })
      },
      children: titlePage()
    },
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: TOP_M, bottom: BOT_M, left: LR_M, right: LR_M, header: 100, footer: 0 }
        }
      },
      headers: { 
        default: makeHeader()
      },
      footers: { 
        default: makeFooter()
      },
      children
    }
  ]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/mnt/user-data/outputs/weather_thesis_beautiful.docx", buf);
  console.log("DONE");
});