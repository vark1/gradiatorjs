declare var Chart: any;

export class LossGraph {
    private chartInstance: any | null = null;
    private ctx: CanvasRenderingContext2D;

    constructor(canvasId: string) {
        const canvas = <HTMLCanvasElement>document.getElementById(canvasId);
        if (!canvas) {
            throw new Error("loss graph canvas not found");
        }
        this.ctx = canvas.getContext('2d')!;
        this.initChart();
    }

    private initChart(): void {
        this.chartInstance = new Chart(this.ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Loss',
                        data: [],
                        borderColor: 'rgba(255,99,132,1)',
                        backgroundColor: 'rgba(255,99,132,0.2)',
                        yAxisID: 'y',
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Accuracy',
                        data: [],
                        borderColor: 'rgba(54,162,235,1)',
                        backgroundColor: 'rgba(54,162,235,0.2)',
                        yAxisID: 'y1',
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Batch Iteration' }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: { display: true, text: 'Loss' },
                        grid: { drawOnChartArea: false }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: { display: true, text: 'Accuracy (%)' },
                        min: 0,
                        max: 100,
                        grid: { drawOnChartArea: true }
                    }
                }
            }
        });
    }

    public addData(iteration: number, loss: number, accuracy: number): void {
        if (!this.chartInstance) return;

        this.chartInstance.data.labels.push(iteration);
        this.chartInstance.data.datasets[0].data.push(loss);
        this.chartInstance.data.datasets[1].data.push(accuracy);

        if (this.chartInstance.data.labels.length % 5 === 0) {
            this.chartInstance.update('none');
        }
    }

    public reset(): void { 
        if (!this.chartInstance) return;
        this.chartInstance.data.labels = [];
        this.chartInstance.data.datasets.forEach((dataset: any)=> {
            dataset.data = [];
        })
        this.chartInstance.update();
    }
}