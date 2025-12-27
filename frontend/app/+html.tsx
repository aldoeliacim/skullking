// Learn more https://docs.expo.dev/router/reference/static-rendering/#root-html

import { ScrollViewStyleReset } from 'expo-router/html';

// This file is web-only and used to configure the root HTML for every
// web page during static rendering.
export default function Root({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
        <title>Skull King</title>
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1, minimum-scale=1, maximum-scale=5, viewport-fit=cover"
        />

        {/* Mobile optimizations */}
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="format-detection" content="telephone=no" />
        <meta name="theme-color" content="#0a1628" />

        {/* PWA manifest */}
        <link rel="manifest" href="/manifest.json" />

        {/* Google Fonts - Pirate themed typography */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Pirata+One&family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&display=swap"
          rel="stylesheet"
        />

        {/* Disable body scrolling on web for better ScrollView behavior */}
        <ScrollViewStyleReset />

        {/* Skull King themed styles */}
        <style
          dangerouslySetInnerHTML={{
            __html: `
              html, body {
                -webkit-text-size-adjust: 100%;
                -moz-text-size-adjust: 100%;
                text-size-adjust: 100%;
                background-color: #0a1628;
              }
              * {
                -webkit-tap-highlight-color: transparent;
                box-sizing: border-box;
              }
              /* Custom scrollbar for dark theme */
              ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
              }
              ::-webkit-scrollbar-track {
                background: #0a1628;
              }
              ::-webkit-scrollbar-thumb {
                background: rgba(212, 168, 75, 0.3);
                border-radius: 4px;
              }
              ::-webkit-scrollbar-thumb:hover {
                background: rgba(212, 168, 75, 0.5);
              }
              /* Selection color */
              ::selection {
                background: rgba(212, 168, 75, 0.3);
                color: #f0ebe3;
              }
            `,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
