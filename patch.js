const Module = require('module');
const path = require('path');
const fs = require('fs');

// Patch fs.writeFileSync to redirect writes from `/mnt/user-data/outputs/...` to current directory
const originalWriteFileSync = fs.writeFileSync;
fs.writeFileSync = function (file, data, options) {
  if (typeof file === 'string' && (file.includes('weather_thesis_beautiful.docx') || file.includes('mnt/user-data') || file.startsWith('/mnt/'))) {
    const filename = path.basename(file);
    const targetPath = path.join(process.cwd(), filename);
    console.log(`[Patch] Redirecting write from "${file}" to "${targetPath}"`);
    return originalWriteFileSync(targetPath, data, options);
  }
  return originalWriteFileSync.apply(this, arguments);
};

const originalRequire = Module.prototype.require;
Module.prototype.require = function (id) {
  if (id === 'docx') {
    const docx = originalRequire.apply(this, arguments);
    if (docx && docx.PageNumber && typeof docx.PageNumber !== 'function') {
      const origPageNumber = docx.PageNumber;
      class PageNumber extends docx.TextRun {
        constructor(options) {
          let typeVal = origPageNumber.CURRENT;
          if (options && options.type === 'total') {
            typeVal = origPageNumber.TOTAL_PAGES;
          }
          super({
            ...options,
            children: [typeVal]
          });
        }
      }
      PageNumber.CURRENT = origPageNumber.CURRENT;
      PageNumber.TOTAL_PAGES = origPageNumber.TOTAL_PAGES;
      PageNumber.CURRENT_SECTION = origPageNumber.CURRENT_SECTION;
      PageNumber.TOTAL_PAGES_IN_SECTION = origPageNumber.TOTAL_PAGES_IN_SECTION;
      
      return new Proxy(docx, {
        get(target, prop) {
          if (prop === 'PageNumber') {
            return PageNumber;
          }
          return target[prop];
        }
      });
    }
  }
  return originalRequire.apply(this, arguments);
};
