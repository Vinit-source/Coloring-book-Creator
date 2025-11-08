import { useCallback } from 'react';

// Since we are loading jsPDF via a script tag, we need to declare it for TypeScript
declare const jspdf: any;

export const usePdfGenerator = () => {
    const downloadPdf = useCallback((images: string[], theme: string, childName: string) => {
        if (!images || images.length === 0) {
            console.error("No images provided to generate PDF.");
            return;
        }

        const { jsPDF } = jspdf;
        const doc = new jsPDF({
            orientation: 'p',
            unit: 'mm',
            format: 'a4'
        });

        // --- Cover Page ---
        const pageHeight = doc.internal.pageSize.getHeight();
        const pageWidth = doc.internal.pageSize.getWidth();
        
        doc.setFillColor(240, 249, 255); // A light blue background
        doc.rect(0, 0, pageWidth, pageHeight, 'F');

        doc.setFont('helvetica', 'bold');
        doc.setFontSize(28);
        doc.setTextColor(25, 25, 112); // Midnight blue
        doc.text(`${childName}'s`, pageWidth / 2, pageHeight / 2 - 20, { align: 'center' });
        
        doc.setFontSize(36);
        doc.setTextColor(255, 105, 180); // Hot pink
        const themeTitle = theme.charAt(0).toUpperCase() + theme.slice(1);
        doc.text(`${themeTitle}`, pageWidth / 2, pageHeight / 2, { align: 'center' });
        
        doc.setFontSize(28);
        doc.setTextColor(25, 25, 112); // Midnight blue
        doc.text(`Coloring Book!`, pageWidth / 2, pageHeight / 2 + 20, { align: 'center' });

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        doc.setTextColor(100, 100, 100);
        doc.text('Generated with AI Coloring Book Creator', pageWidth / 2, pageHeight - 10, { align: 'center' });


        // --- Coloring Pages ---
        images.forEach((base64Img) => {
            doc.addPage();
            const imgData = `data:image/png;base64,${base64Img}`;
            
            // A4 page is 210x297 mm. We'll add some margin.
            const margin = 10;
            const imgWidth = pageWidth - (margin * 2);
            const imgHeight = (imgWidth / 4) * 3; // Maintain 4:3 aspect ratio

            const x = margin;
            const y = (pageHeight - imgHeight) / 2; // Center vertically

            doc.addImage(imgData, 'PNG', x, y, imgWidth, imgHeight);
            
            // Add a simple border
            doc.setDrawColor(200, 200, 200);
            doc.rect(margin / 2, margin / 2, pageWidth - margin, pageHeight - margin);
        });

        // --- Download ---
        const safeFileName = theme.replace(/[^a-z0-9]/gi, '_').toLowerCase();
        doc.save(`${childName}_${safeFileName}_coloring_book.pdf`);
    }, []);

    return { downloadPdf };
};
