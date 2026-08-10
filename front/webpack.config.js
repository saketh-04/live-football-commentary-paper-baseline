const path = require('path');
const fs = require('fs');

module.exports = {
    mode: 'development',
    entry: './src/app.js',
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname)
    },
    module: {
        rules: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use: {
                    loader: 'babel-loader',
                    options: {
                        presets: ['@babel/preset-env']
                    }
                }
            }
        ]
    },
    plugins: [
        function() {
            this.hooks.emit.tapAsync('ReadVideoPathsPlugin', (compilation, callback) => {
                fs.readFile('local_video_paths.txt', 'utf8', (err, data) => {
                    if (err) {
                        throw err;
                    }
                    const videoPaths = data.split('\n').filter(Boolean);
                    const videoPathsModule = `export const videoPaths = ${JSON.stringify(videoPaths)};`;
                    fs.writeFileSync(path.resolve(__dirname, 'src', 'videoPaths.js'), videoPathsModule);
                    callback();
                });
            });
        }
    ]
};

