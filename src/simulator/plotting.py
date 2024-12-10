from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import plotly.graph_objects as go
import numpy as np
from copy import copy
from plotly.subplots import make_subplots

from .slot_state import SlotState
from .request import Request
from .metrics import Metrics

if TYPE_CHECKING:
    from .engine import Engine


@dataclass
class PlotDataEntry:
    slot_state: SlotState = SlotState.empty
    request: Optional[Request] = None


class PlotData:
    num_slots: int
    slots_contents: List[List[PlotDataEntry]]  # slots_contents[slot][time]
    engine: "Engine"
    metrics: Metrics

    def __init__(self, num_slots, engine):
        self.num_slots = num_slots
        self.slots_contents = [[] for _ in range(num_slots)]
        self.engine = engine
        self.metrics = Metrics(num_slots, engine)

    def track_previous_batch(self):
        self.metrics.track_previous_batch()

    def track_current_batch(self):
        data = []
        for s in self.engine.get_all_slots():
            if s in self.engine.get_occupied_slots():
                pde = PlotDataEntry(
                    request=copy(self.engine.current_batch[s]),
                    slot_state=self.engine.current_batch[s].get_slot_state_at(
                        self.engine.current_time
                    ),
                )
            else:
                pde = PlotDataEntry(request=None, slot_state=SlotState.empty)
            data.append(pde)
        self._add_batch(
            data, self.engine.current_time + self.engine.get_current_batch_duration()
        )
        self.metrics.track_current_batch()

    def _add_batch(self, batch, end_time):
        for slot, pd in zip(self.slots_contents, batch):
            slot.append(pd)
        self.metrics.times.append(end_time)

    def get_plot_z(self) -> List[List[int]]:
        return list([s.slot_state.value for s in sc] for sc in self.slots_contents)

    def get_plot_annotations(self) -> List[go.layout.Annotation]:
        annotations = []
        for slot_id, slot_contents in enumerate(self.slots_contents):
            for time, plot_data_entry in enumerate(slot_contents):
                annotations.append(
                    go.layout.Annotation(
                        text=plot_data_entry.slot_state.name,
                        showarrow=False,
                        x=time,
                        y=slot_id,
                    )
                )
        return annotations

    def get_plot_text(self) -> List[List[str]]:
        text = []
        for slot_id, slot_contents in enumerate(self.slots_contents):
            row = []
            for time, plot_data_entry in enumerate(slot_contents):
                row.append(
                    plot_data_entry.slot_state.name[0],
                )
            text.append(row)
        return text

    def get_plot_customdata(self) -> List[List[str]]:
        customdata = []
        for slot_id, slot_contents in enumerate(self.slots_contents):
            row = []
            for time_id, plot_data_entry in enumerate(slot_contents):
                ti = self.metrics.get_time_interval(time_id)
                time_interval_str = (
                    f"{ti[0]:.1f}-{ti[1]:.1f}, ({ti[1]-ti[0]:.1f} ticks)"
                )
                if plot_data_entry.request is None:
                    row.append(
                        [
                            ""
                            + f"Empty slot: {slot_id}<br>"
                            + f"Time: {time_interval_str}<br>"
                            + f"Queue size: {self.metrics.queue_size[time_id]}<br>"
                        ]
                    )
                else:
                    row.append(
                        [
                            ""
                            + f"Stage: {plot_data_entry.slot_state.name}<br>"
                            + f"Current token #: {plot_data_entry.request.tokens_generated}<br>"
                            + f"Current latency: {plot_data_entry.request.get_current_latency_at(self.metrics.times[time_id]):.1f} ticks<br>"
                            + f"Request id: {plot_data_entry.request.id}<br>"
                            + f"Time: {time_interval_str}<br>"
                            + f"Queue size: {self.metrics.queue_size[time_id]}<br>"
                        ]
                    )
            customdata.append(row)
        return customdata

    def render(self):
        engine = self.engine
        # Create the figure
        heatmap = go.Heatmap(
            z=engine.plot_data.get_plot_z(),
            customdata=engine.plot_data.get_plot_customdata(),
            x=engine.plot_data.metrics.times,
            y=np.arange(engine.max_batch_size),
            # showlegend=True,
            showscale=False,
            text=engine.plot_data.get_plot_text(),
            hovertext=engine.plot_data.get_plot_customdata(),
            hovertemplate="%{customdata[0]}<extra></extra>",
            texttemplate="%{text}",
            xgap=1,
            ygap=1,
            zmin=0,
            zmax=2,
        )

        queue_size = go.Scatter(
            x=engine.plot_data.metrics.times[1:],
            y=engine.plot_data.metrics.queue_size,
            name="Queue Size",
            hovertemplate="Queue Size: %{y} at %{x} ticks<extra></extra>",
            yaxis="y2",
            mode="lines",
            line=dict(color="orange", width=4),
        )

        e2e_latency = go.Scatter(
            x=[t for t, latency in engine.plot_data.metrics.e2e_latency],
            y=[latency for t, latency in engine.plot_data.metrics.e2e_latency],
            name="E2E Latency",
            hovertemplate="E2E latency: %{y} at %{x} ticks<extra></extra>",
            yaxis="y3",
            mode="markers",
            marker=dict(
                color="greenyellow", size=10, line=dict(color="White", width=2)
            ),
        )

        ttft = go.Scatter(
            x=[t for t, latency in engine.plot_data.metrics.ttft],
            y=[latency for t, latency in engine.plot_data.metrics.ttft],
            name="TTFT",
            hovertemplate="TTFT: %{y} at %{x} ticks<extra></extra>",
            yaxis="y3",
            mode="markers",
            marker=dict(color="deepPink", size=10, line=dict(color="White", width=2)),
        )

        # fig.update_layout(annotations=engine.plot_data.get_plot_annotations())

        # Combine the figures using make_subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.2],
            vertical_spacing=0.15,
            specs=[
                [{}],
                [dict(secondary_y=True)],
            ],
        )
        fig.add_trace(heatmap, row=1, col=1)
        fig.add_trace(queue_size, row=2, col=1, secondary_y=True)
        fig.add_trace(
            e2e_latency,
            row=2,
            col=1,
        )
        fig.add_trace(ttft, row=2, col=1)

        # Customize the layout

        fig.update_xaxes(title="Time, ticks")
        fig.update_xaxes(
            row=1,
            col=1,
            showgrid=False,
            zeroline=False,
            ticks="outside",
            showticklabels=True,
        )
        fig.update_xaxes(row=2, col=1, side="top", title="")
        fig.update_yaxes(
            row=1,
            col=1,
            title="Slot",
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            row=2,
            col=1,
            title="Latency, ticks",
        )
        fig.update_yaxes(
            row=2,
            col=1,
            secondary_y=True,
            title="Queue Size",
            showgrid=False,
            zeroline=False,
        )

        fig.update_layout(
            title="Prefill and Decoding Simulation",
            height=600,
            plot_bgcolor="black",
        )
        return fig

    def show(self, fig=None):
        fig = fig or self.render()
        fig.show()

    def save(self, filename="temp.png", fig=None):
        fig = fig or self.render()
        fig.write_image(filename, format=filename[filename.index(".") + 1 :])
        return filename
