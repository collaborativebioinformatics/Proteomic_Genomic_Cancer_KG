'use strict';

var obsidian = require('obsidian');
var electron = require('electron');
var path_1 = require('path');

function _interopDefaultLegacy (e) { return e && typeof e === 'object' && 'default' in e ? e : { 'default': e }; }

var path_1__default = /*#__PURE__*/_interopDefaultLegacy(path_1);

/*! *****************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
/* global Reflect, Promise */

var extendStatics = function(d, b) {
    extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
    return extendStatics(d, b);
};

function __extends(d, b) {
    if (typeof b !== "function" && b !== null)
        throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
    extendStatics(d, b);
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}

function __awaiter(thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
}

function __generator(thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
}

/**
 * File URI to Path function.
 *
 * @param {String} uri
 * @return {String} path
 * @api public
 */
function fileUriToPath(uri) {
    if (typeof uri !== 'string' ||
        uri.length <= 7 ||
        uri.substring(0, 7) !== 'file://') {
        throw new TypeError('must pass in a file:// URI to convert to a file path');
    }
    const rest = decodeURI(uri.substring(7));
    const firstSlash = rest.indexOf('/');
    let host = rest.substring(0, firstSlash);
    let path = rest.substring(firstSlash + 1);
    // 2.  Scheme Definition
    // As a special case, <host> can be the string "localhost" or the empty
    // string; this is interpreted as "the machine from which the URL is
    // being interpreted".
    if (host === 'localhost') {
        host = '';
    }
    if (host) {
        host = path_1__default['default'].sep + path_1__default['default'].sep + host;
    }
    // 3.2  Drives, drive letters, mount points, file system root
    // Drive letters are mapped into the top of a file URI in various ways,
    // depending on the implementation; some applications substitute
    // vertical bar ("|") for the colon after the drive letter, yielding
    // "file:///c|/tmp/test.txt".  In some cases, the colon is left
    // unchanged, as in "file:///c:/tmp/test.txt".  In other cases, the
    // colon is simply omitted, as in "file:///c/tmp/test.txt".
    path = path.replace(/^(.+)\|/, '$1:');
    // for Windows, we need to invert the path separators from what a URI uses
    if (path_1__default['default'].sep === '\\') {
        path = path.replace(/\//g, '\\');
    }
    if (/^.+:/.test(path)) ;
    else {
        // unix path…
        path = path_1__default['default'].sep + path;
    }
    return host + path;
}
var src = fileUriToPath;

var FilePathToUri = /** @class */ (function (_super) {
    __extends(FilePathToUri, _super);
    function FilePathToUri() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    FilePathToUri.prototype.onload = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            return __generator(this, function (_a) {
                console.log('Loading plugin FilePathToUri...');
                this.addCommand({
                    id: 'toggle-file-path-to-uri',
                    name: 'Toggle selected file path to URI and back',
                    checkCallback: function (checking) {
                        if (_this.getEditor() === null) {
                            return;
                        }
                        if (!checking) {
                            _this.toggleLink();
                        }
                        return true;
                    },
                    hotkeys: [
                        {
                            modifiers: ['Mod', 'Alt'],
                            key: 'L',
                        },
                    ],
                });
                this.addCommand({
                    id: 'paste-file-path-as-file-uri',
                    name: 'Paste file path as file uri',
                    checkCallback: function (checking) {
                        if (_this.getEditor() === null) {
                            return;
                        }
                        if (!checking) {
                            _this.pasteAsUri();
                        }
                        return true;
                    },
                    hotkeys: [
                        {
                            modifiers: ['Mod', 'Alt', 'Shift'],
                            key: 'L',
                        },
                    ],
                });
                this.addCommand({
                    id: 'paste-file-path-as-file-uri-link',
                    name: 'Paste file path as file uri link',
                    checkCallback: function (checking) {
                        if (_this.getEditor() === null) {
                            return;
                        }
                        if (!checking) {
                            _this.pasteAsUriLink();
                        }
                        return true;
                    },
                    hotkeys: [],
                });
                this.addCommand({
                    id: 'paste-file-path-as-file-uri-link-name-only',
                    name: 'Paste file path as file uri link - Name only',
                    checkCallback: function (checking) {
                        if (_this.getEditor() === null) {
                            return;
                        }
                        if (!checking) {
                            _this.pasteAsUriLink('lastSegment');
                        }
                        return true;
                    },
                    hotkeys: [
                    // For testing only
                    // {
                    // 	modifiers: ['Mod', 'Alt', 'Shift'],
                    // 	key: 'J',
                    // },
                    ],
                });
                return [2 /*return*/];
            });
        });
    };
    FilePathToUri.prototype.getEditor = function () {
        var view = this.app.workspace.getActiveViewOfType(obsidian.MarkdownView);
        if (!view || view.getMode() !== 'source') {
            return null;
        }
        return view.sourceMode.cmEditor;
    };
    FilePathToUri.prototype.pasteAsUri = function () {
        var editor = this.getEditor();
        if (editor == null) {
            return;
        }
        var clipboardText = electron.clipboard.readText('clipboard');
        if (!clipboardText) {
            return;
        }
        clipboardText = this.cleanupText(clipboardText);
        // Paste the text as usual if it's not file path
        if (clipboardText.startsWith('file:') || !this.hasSlashes(clipboardText)) {
            editor.replaceSelection(clipboardText, 'around');
        }
        // network path '\\path'
        if (clipboardText.startsWith('\\\\')) {
            var endsWithSlash = clipboardText.endsWith('\\') || clipboardText.endsWith('/');
            // URL throws error on invalid url
            try {
                var url = new URL(clipboardText);
                var link = url.href.replace('file://', 'file:///%5C%5C');
                if (link.endsWith('/') && !endsWithSlash) {
                    link = link.slice(0, -1);
                }
                editor.replaceSelection(link, 'around');
            }
            catch (e) {
                return;
            }
        }
        // path C:\Users\ or \System\etc
        else {
            if (!this.hasSlashes(clipboardText)) {
                return;
            }
            // URL throws error on invalid url
            try {
                var url = new URL('file://' + clipboardText);
                editor.replaceSelection(url.href, 'around');
            }
            catch (e) {
                return;
            }
        }
    };
    FilePathToUri.prototype.makeLink = function (title, link) {
        return "[" + title.replace(new RegExp('%20', 'g'), ' ') + "](" + link + ")";
    };
    // TODO: todo
    FilePathToUri.prototype.pasteAsUriLink = function (pasteType) {
        var editor = this.getEditor();
        if (editor == null) {
            return;
        }
        var clipboardText = electron.clipboard.readText('clipboard');
        if (!clipboardText) {
            return;
        }
        clipboardText = this.cleanupText(clipboardText);
        // Paste the text as usual if it's not file path
        if (clipboardText.startsWith('file:') || !this.hasSlashes(clipboardText)) {
            editor.replaceSelection(clipboardText, 'around');
        }
        // network path '\\path'
        if (clipboardText.startsWith('\\\\')) {
            var endsWithSlash = clipboardText.endsWith('\\') || clipboardText.endsWith('/');
            // URL throws error on invalid url
            try {
                var url = new URL(clipboardText);
                var link = url.href.replace('file://', 'file:///%5C%5C');
                if (link.endsWith('/') && !endsWithSlash) {
                    link = link.slice(0, -1);
                }
                // https://stackoverflow.com/questions/8376525/get-value-of-a-string-after-last-slash-in-javascript
                // Need to use url.href cause it normalizes the slash type (\/)
                // Handles trailing slash
                var lastUrlSegment = /[^/]*$/.exec(url.href.endsWith('/') ? url.href.slice(0, -1) : url.href)[0];
                if (pasteType === 'lastSegment') {
                    editor.replaceSelection(this.makeLink(lastUrlSegment, link), 'around');
                }
                else {
                    // Needs to add two '\\' (that is '\\\\' in code because of escaping) in order for the link title
                    // to display two '\\' in preview mode
                    editor.replaceSelection(this.makeLink('\\\\' + clipboardText, link), 'around');
                }
            }
            catch (e) {
                return;
            }
        }
        // path C:\Users\ or \System\etc
        else {
            if (!this.hasSlashes(clipboardText)) {
                return;
            }
            // URL throws error on invalid url
            try {
                var url = new URL('file://' + clipboardText);
                // https://stackoverflow.com/questions/8376525/get-value-of-a-string-after-last-slash-in-javascript
                // Need to use url.href cause it normalizes the slash type (\/)
                // Handles trailing slash
                var lastUrlSegment = /[^/]*$/.exec(url.href.endsWith('/') ? url.href.slice(0, -1) : url.href)[0];
                // Ugly, but quick solution
                if (pasteType === 'lastSegment') {
                    editor.replaceSelection(this.makeLink(lastUrlSegment, url.href), 'around');
                }
                else {
                    editor.replaceSelection(this.makeLink(clipboardText, url.href), 'around');
                }
            }
            catch (e) {
                return;
            }
        }
    };
    /**
     * Does the text have any '\' or '/'?
     */
    FilePathToUri.prototype.hasSlashes = function (text) {
        // Does it have any '\' or '/'?
        var regexHasAnySlash = /.*([\\\/]).*/g;
        if (typeof text !== 'string') {
            return false;
        }
        var matches = text.match(regexHasAnySlash);
        return !!matches;
    };
    /**
     * Trim whitespace and remove surrounding "
     */
    FilePathToUri.prototype.cleanupText = function (text) {
        if (typeof text !== 'string') {
            return '';
        }
        text = text.trim();
        // Remove surrounding "
        if (text.startsWith('"')) {
            text = text.substr(1);
        }
        if (text.endsWith('"')) {
            text = text.substr(0, text.length - 1);
        }
        return text;
    };
    FilePathToUri.prototype.toggleLink = function () {
        var editor = this.getEditor();
        if (editor == null || !editor.somethingSelected()) {
            return;
        }
        var selectedText = editor.getSelection();
        selectedText = this.cleanupText(selectedText);
        // file url for network location file://\\location
        // Works for both 'file:///\\path' and 'file:///%5C%5Cpath'
        // Obsidian uses escape chars in link so `file:///\\location` will try to open `file:///\location instead
        // But the selected text we get contains the full string, thus the test for both 2 and 4 '\' chars
        if (selectedText.startsWith('file:///\\\\') ||
            selectedText.startsWith('file:///\\\\\\\\') ||
            selectedText.startsWith('file:///%5C%5C')) {
            // normalize to 'file:///'
            selectedText = selectedText.replace('file:///\\\\\\\\', 'file:///');
            selectedText = selectedText.replace('file:///\\\\', 'file:///');
            selectedText = selectedText.replace('file:///%5C%5C', 'file:///');
            var url = src(selectedText);
            if (url) {
                // fileUriToPath returns only single leading '\' so we need to add the second one
                editor.replaceSelection('\\' + url, 'around');
            }
        }
        // file link file:///C:/Users
        else if (selectedText.startsWith('file:///')) {
            var url = src(selectedText);
            if (url) {
                editor.replaceSelection(url, 'around');
            }
        }
        // network path '\\path'
        else if (selectedText.startsWith('\\\\')) {
            var endsWithSlash = selectedText.endsWith('\\') || selectedText.endsWith('/');
            // URL throws error on invalid url
            try {
                var url = new URL(selectedText);
                var link = url.href.replace('file://', 'file:///%5C%5C');
                if (link.endsWith('/') && !endsWithSlash) {
                    link = link.slice(0, -1);
                }
                editor.replaceSelection(link, 'around');
            }
            catch (e) {
                return;
            }
        }
        // path C:\Users\ or \System\etc
        else {
            if (!this.hasSlashes(selectedText)) {
                return;
            }
            // URL throws error on invalid url
            try {
                var url = new URL('file://' + selectedText);
                editor.replaceSelection(url.href, 'around');
            }
            catch (e) {
                return;
            }
        }
    };
    FilePathToUri.prototype.onunload = function () {
        console.log('Unloading plugin FilePathToUri...');
    };
    return FilePathToUri;
}(obsidian.Plugin));

module.exports = FilePathToUri;


/* nosourcemap */