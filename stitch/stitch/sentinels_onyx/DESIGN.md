# Design System Strategy: High-End Fintech & AI Fraud Detection

## 1. Overview & Creative North Star: "The Vigilant Lens"

This design system is built to convey more than just security; it communicates **predictive intelligence**. Moving beyond the "standard SaaS" aesthetic, we are adopting a Creative North Star titled **"The Vigilant Lens."** 

The interface should feel like a high-performance instrument—cinematic, deep, and hyper-focused. We break the traditional grid through intentional asymmetry: large, editorial headlines paired with compact, data-dense modules. By utilizing layered glassmorphism and grain textures, we create a sense of physical depth, making the user feel they are looking *into* the data rather than just looking *at* a screen.

---

## 2. Colors & Surface Philosophy

Our palette is anchored in deep obsidian and navy tones, punctuated by high-chroma functional signals. The system now operates in **light mode**, providing a fresh perspective while retaining its core principles.

### The "No-Line" Rule
Standard UI relies on borders to separate content. This system prohibits 1px solid borders for sectioning. Boundaries must be defined strictly through background color shifts. 
*   Use `surface-container-low` (#1A1C20) against `background` (#111318) to create structural zones.
*   Transitions should feel organic, moving from the void (black) to functional surfaces (deep navy).

### Surface Hierarchy & Nesting
Treat the UI as a physical stack of frosted glass.
*   **Base:** `surface-dim` (#111318) with a subtle animated grain overlay.
*   **Intermediate:** `surface-container` (#1E2024) for secondary content blocks.
*   **Active/Floating:** `surface-container-highest` (#333539) for cards and modals.
*   **The Glass & Gradient Rule:** For hero sections and primary CTAs, use a linear gradient from `primary` (#ADC6FF) to `primary_container` (#4B8FF). This adds "visual soul" and prevents the interface from feeling flat or sterile.

---

## 3. Typography: The Editorial Authority

We use a dual-typeface system to balance high-end editorial style with technical precision.

*   **Display & Headlines (Manrope):** Chosen for its geometric precision and modern weight. Use `display-lg` (3.5rem) with tight letter-spacing (-0.02em) for landing pages to create a "Brex-style" authoritative presence.
*   **Body & Labels (Inter):** The workhorse for data. Inter provides maximum legibility at small sizes (`body-sm` at 0.75rem).
*   **Hierarchy as Brand:** Use extreme scale contrast. A `display-md` headline should sit confidently next to a `label-md` metadata point. This contrast suggests that while the system handles massive AI computations, it remains obsessed with the smallest details of fraud.

---

## 4. Elevation & Depth: Tonal Layering

Shadows and borders in this system are "atmospheric" rather than structural.

*   **The Layering Principle:** Instead of traditional shadows, stack surfaces. Place a `surface-container-lowest` card inside a `surface-container-high` section to create a "recessed" or "inset" look.
*   **Ambient Shadows:** When an element must float (e.g., a dropdown), use a shadow color tinted with `on-surface` (#E2E2E8) at 4% opacity. The blur radius should be large (30px+) to mimic natural light dispersion.
*   **The "Ghost Border" Fallback:** If accessibility requires a border, use the `outline-variant` (#414755) at 10% opacity. This creates a "glow" effect rather than a hard line.
*   **Glassmorphism:** Apply a `backdrop-filter: blur(12px)` to any element using `surface-container-highest`. This allows the deep navy background gradient to bleed through, softening the edges of the container.

---

## 5. Components

### Buttons
*   **Primary:** Background `primary_container` (#4B8FF), Text `on_primary` (#002E69). No border. High corner radius (`lg`: 0.5rem).
*   **Secondary:** Ghost style. No background, `outline-variant` at 20% opacity for the border.
*   **Tertiary:** Text-only using `primary` (#ADC6FF) with a hover state that shifts the background to `surface-bright` (#37393E).

### Glassmorphic Cards
*   **Background:** `surface-container-highest` at 60% opacity.
*   **Border:** 1px solid `on-surface` at 10% opacity (The "Ghost Border").
*   **Content:** No divider lines. Separate data points using the `xl` (0.75rem) or `lg` (0.5rem) spacing scale.

### Input Fields
*   **Resting:** `surface-container-low` background. No border.
*   **Focus:** A 1px "Ghost Border" using `primary`. 
*   **Error:** Use `tertiary_container` (#FF5545) for the background shift and `tertiary` (#FFB4AA) for the label text.

### Detection Chips (SaaS Specific)
*   **Safe:** `secondary_container` background with `on_secondary` text.
*   **At Risk:** `error_container` background with `on_error` text.
*   These should be pill-shaped (`full` roundedness) to contrast against the more architectural card shapes.

---

## 6. Do’s and Don’ts

### Do
*   **Do** use extreme whitespace. If a layout feels "busy," increase the padding to the next tier in the spacing scale.
*   **Do** use subtle motion. Background grain and glass blurs should feel alive, not static.
*   **Do** use `primary_fixed` (#D8E2FF) for secondary text within high-importance modules to maintain a monochromatic, premium feel.

### Don't
*   **Don't** use 100% opaque white borders. It breaks the "Vigilant Lens" immersion.
*   **Don't** use standard grey shadows. Always tint shadows with the background color to avoid a "dirty" look.
*   **Don't** use divider lines to separate list items. Use background color alternates or simple vertical padding.
*   **Don't** use "Electric Blue" for anything other than primary actions. Overusing it diminishes its power as a conversion tool.